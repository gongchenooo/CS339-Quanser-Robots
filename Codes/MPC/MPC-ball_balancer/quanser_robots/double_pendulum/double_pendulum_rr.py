import numpy as np
import time
from quanser_robots.common import QSocket, VelocityFilter
from quanser_robots.double_pendulum.base import DoublePendulumBase
from quanser_robots.double_pendulum.ctrl import GoToLimCtrl


class DoublePendulum(DoublePendulumBase):
    def __init__(self, ip, fs_ctrl):
        super(DoublePendulum, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)

        # Initialize Socket:
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0], u_len=self.action_space.shape[0])
        self._sens_offset = None

        # Save the relative limits:
        self._calibrated = False
        self._norm_x_lim = np.zeros(2, dtype=np.float32)
        self.c_lim = 0.075

    def __del__(self):
        if self._qsoc.is_open():
            self.close()

    def _calibrate(self, verbose=False):
        if self._calibrated:
            return

        # Retrieve current state:
        sensor = self._qsoc.snd_rcv(np.array([0.0]))
        sensor[1] -= np.pi

        # Reset calibration
        wcf = 62.8318
        zetaf = 0.9

        self._vel_filt_x   = VelocityFilter(2, num=(wcf**2, 0), den=(1, 2*wcf*zetaf, wcf**2), x_init=sensor[0:1], dt=self.timing.dt)
        self._vel_filt_th0 = VelocityFilter(2, num=(wcf**2, 0), den=(1, 2*wcf*zetaf, wcf**2), x_init=sensor[1:2], dt=self.timing.dt)
        self._vel_filt_th1 = VelocityFilter(2, num=(wcf**2, 0), den=(1, 2*wcf*zetaf, wcf**2), x_init=sensor[2:3], dt=self.timing.dt)
        self._sens_offset = np.zeros(self.sensor_space.shape[0], dtype=np.float32)

        # Go to the left:
        if verbose:
            print("\tGo to the Left:\t\t\t", end="")

        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=True)

        while not ctrl.done :
            a = ctrl(state)
            state = self._sim_step(a)

        if ctrl.success:
            self._norm_x_lim[1] = state[0]
            if verbose: print("\u2713")

        else:
            if verbose: print("\u274C ")
            raise RuntimeError("Going to the left limit failed.")

        # Go to the right
        if verbose:
            print("\tGo to the Right:\t\t", end="")

        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=False)

        while not ctrl.done :
            a = ctrl(state)
            state = self._sim_step(a)

        if ctrl.success:
            self._norm_x_lim[0] = state[0]
            if verbose: print("\u2713")

        else:
            if verbose: print("\u274C ")
            raise RuntimeError("Going to the right limit failed.")

        # Activate the absolute cart position:
        self._calibrated = True

    def _center_cart(self, verbose=False):
        t_max = 10.0

        if verbose:
            print("\tCentering the Cart:\t\t", end="")

        # Center the cart:
        t0 = time.time()
        state = self._zero_sim_step()
        while (time.time() - t0) < t_max:
            a = -np.sign(state[0]) * 1.5 * np.ones(1)
            state = self._sim_step(a)

            if np.abs(state[0]) <= self.c_lim/10.:
                break

        # Stop the Cart:
        state = self._zero_sim_step()
        time.sleep(0.5)

        if np.abs(state[0]) > self.c_lim:
            if verbose: print("\u274C")
            time.sleep(0.1)
            raise RuntimeError("Centering of the cart failed. |x| = {0:.2f} > {1:.2f}".format(np.abs(state[0]), self.c_lim))

        elif verbose:
            print("\u2713")

    def _wait_for_upright_pole(self, verbose=False):
        if verbose:
            print("\tCentering the Pole:\t\t", end="")

        t_max = 15.0
        upright = False

        pos_th = np.array([self.c_lim, 2. * np.pi / 180., 2. * np.pi / 180.])
        vel_th = 0.1 * np.ones(3)
        th = np.hstack((pos_th, vel_th))

        # Wait until the pole is upright:
        t0 = time.time()
        while (time.time() - t0) <= t_max:
            state = self._zero_sim_step()
            #time.sleep(1./550.)

            if np.all(np.abs(state) <= th):
                upright = True
                break

        if not upright:
            if verbose: print("\u274C")
            time.sleep(0.1)

            state_str = np.array2string(np.abs(state), suppress_small=True, precision=2, formatter={'float_kind': lambda x: "{0:+05.2f}".format(x)})
            th_str = np.array2string(th, suppress_small=True, precision=2, formatter={'float_kind': lambda x: "{0:+05.2f}".format(x)})
            raise TimeoutError("The Pole is not upright, i.e., {0} > {1}".format(state_str, th_str))

        elif verbose:
            print("\u2713")

        return

    def _sim_step(self, a):
        pos = self._qsoc.snd_rcv(a)

        # Transform the relative cart position to [-0.4, +0.4]
        if self._calibrated:
            pos[0] = (pos[0] - self._norm_x_lim[0]) - 1./2. * (self._norm_x_lim[1] - self._norm_x_lim[0])

        # Transform theta_0, such that 0.0 means an upright pole:
        pos[1] -= np.pi

        x_dot = self._vel_filt_x(pos[0:1])
        th0_dot = self._vel_filt_th0(pos[1:2])
        th1_dot = self._vel_filt_th1(pos[2:3])

        # Normalize the angle from -pi to +pi:
        pos[1] = np.mod(pos[1] + np.pi, 2. * np.pi) - np.pi
        return np.concatenate([pos, x_dot, th0_dot, th1_dot])

    def reset(self, verbose=True):
        if verbose:
            print("\nReset Double Pendulum:")

        # Reconnect to the system:
        self._qsoc.close()
        self._qsoc.open()

        # The system only needs to be calibrated once, as this is a bit time consuming:
        self._calibrate(verbose=verbose)

        # Center the cart in the middle @ x = 0.0
        self._center_cart(verbose=verbose)

        # Wait until the Pole is upright:
        self._wait_for_upright_pole(verbose=verbose)

        self._state = self._zero_sim_step()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
