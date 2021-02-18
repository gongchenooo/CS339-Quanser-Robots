import numpy as np

from .base import BallBalancerBase, BallBalancerDynamics
from ..common import QSocket, VelocityFilter


class BallBalancerRR(BallBalancerBase):
    """
    Quanser 2 DoF Ball Balancer real robot class.
    """
    def __init__(self, fs_ctrl, ip="130.83.164.52", simplified_dyn=False, wait_for_input=False):
        super().__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._dyn = BallBalancerDynamics(dt=self.timing.dt, simplified_dyn=simplified_dyn)

        # Initialize communication
        self._qsoc = QSocket(ip, self.sensor_space.shape[0], self.action_space.shape[0])

        self._tol = 0.  # disable tolerance for done flag
        self.wait_for_input = wait_for_input  # pause within the reset function to manipulate the device

    def reset(self):
        super().reset()
        # Cancel and re-open the connection to the socket
        self._qsoc.close()
        if self.wait_for_input:
            input()
        self._qsoc.open()

        # Initialize velocity filter
        # Send actions and receive sensor measurements. One extra send & receive for initializing the filter.
        pos_meas = self._qsoc.snd_rcv(np.zeros(self.action_space.shape))
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0],
                                        dt=self.timing.dt,
                                        x_init=pos_meas)

        # Start gently with a zero action
        obs, _, _, _ = self.step(np.zeros(self.action_space.shape))

        return obs

    def step(self, action):
        """
        Send command and receive next state.
        """
        assert action is not None, "Action should be not None"
        assert isinstance(action, np.ndarray), "The action should be a ndarray"
        assert not np.isnan(action).any(), "Action NaN is not a valid action"
        assert action.ndim == 1, "The action = {1} must be 1d but the input is {0:d}d".format(action.ndim, action)

        info = {'action_raw': action}
        # Apply action limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._curr_action = action

        # Send actions and receive sensor measurements
        pos_meas = self._qsoc.snd_rcv(action)

        # Construct the state from measurements and observer (filter)
        obs = np.r_[pos_meas, self._vel_filt(pos_meas)]
        self._state = obs

        reward = self._rew_fcn(obs, action)
        done = self._is_done()  # uses the state estimated from measurements

        self._step_count += 1
        return obs, reward, done, info

    def render(self, mode='human', render_step=10):
        # Cheap printing to console
        if self._step_count % render_step == 0:
            print("time step: {:3}  |  in bounds: {:1}  |  state: {}  |  action: {}".format(
                self._step_count, self.state_space.contains(self._state), self._state, self._curr_action))

            if not self.state_space.contains(self._state):
                # State is out of bounds
                np.set_printoptions(precision=3)
                print("min state : ", self.state_space.low)
                print("last state: ", self._state)
                print("max state : ", self.state_space.high)

    def close(self):
        # Terminate gently with a zero action
        self.step(np.array([0.0, 0.0]))

        # Cancel the connection to the socket
        self._qsoc.close()


if __name__ == "__main__":
    bb = BallBalancerRR(fs_ctrl=500)
    s = bb.step(np.array([0.0, 0.0]))
    print("state: ", s)
