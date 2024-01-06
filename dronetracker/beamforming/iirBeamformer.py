import numpy as np
from numpy import sin, cos, pi, exp
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq, ifft
from scipy.spatial.transform import Rotation as R


class IirBeamFormer:
    filterbank = None

    i = 0

    def __init__(self, block_len, f_low, f_high, f_s, v_m):
        self._block_len = block_len
        self._f_low = int(2 * block_len / f_s * f_low)
        self._f_high = int(2 * block_len / f_s * f_high)
        self._fs = f_s
        self._vm = v_m
        self.audio = []
        pass

    def compute_angled_filterbank(self, coord, phi, theta):
        angles = np.vstack((-phi, pi / 2 - theta)).reshape(-1, 2)
        angles = np.vstack((-phi, pi / 2 - theta)).swapaxes(0, 1)
        #         angles = np.vstack((pi/2 - theta,-phi)).reshape(-1,2)
        #         import ipdb; ipdb.set_trace()

        mat = R.from_euler("zy", angles, degrees=False)
        coord_t = (mat.as_matrix() @ coord.T).swapaxes(-1, -2)
        #         coord_t = coord_t.swapaxes(1,0)
        x = coord_t[:, :, 0]
        y = coord_t[:, :, 1]
        r_m = np.sqrt(x**2 + y**2)[..., np.newaxis]
        phi_m = np.arctan2(y, x)[..., np.newaxis]
        f = (
            np.arange(self._block_len).reshape(1, 1, -1)
            * self._fs
            / (2 * self._block_len)
        )

        t_s = sin(pi / 2 - phi_m) * r_m * (1 / self._vm)
        fb = np.zeros((*np.squeeze(t_s).shape, self._block_len), dtype=np.complex64)
        fb[:, :, self._f_low : self._f_high] = exp(
            -1j * 2 * pi * f[:, :, self._f_low : self._f_high] * (t_s)
        )

        self.filterbank = fb
        self.minFb = self.filterbank[:, :, self._f_low : self._f_high]
        return fb

    def global_beam_sweep(self, block):
        #         self.audio.append(block)
        tracks = fft(block.T, axis=-1)
        tracks = tracks[: self._block_len]
        print("calc...")
        fb_a = self.minFb * tracks[:, self._f_low : self._f_high]
        #         print(fb_a.shape)
        #     fb = fb_a * tracks
        #     response = np.sum(np.abs(np.sum(fb, axis=-2)[:, :, 50:200]) ** 2, axis=-1)
        signals = np.abs(np.sum(fb_a, axis=-2))  # [:, :, 25:100])
        response = np.sum(signals**2, axis=-1) / (self._f_high - self._f_low)
        return response

    def local_beam_sweep(self, signals, center_direction):
        pass

    def listen_at(self, signals, direction):
        pass
