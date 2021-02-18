import numpy as np
from ..common import QSocket, VelocityFilter
from .base import LevitationBase


class Levitation(LevitationBase):
    def __init__(self, ip, fs_ctrl):
        super(Levitation, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0],
                             u_len=self.action_space.shape[0])
        self._sens_offset = None

    def _calibrate(self):
        # Reset calibration
        # Set current state
        pass
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        pos = self._qsoc.snd_rcv(a)
        pos -= self._sens_offset
        return np.concatenate([pos, self._vel_filt(pos)])

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._calibrate()
        return self.step([0.0])[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
