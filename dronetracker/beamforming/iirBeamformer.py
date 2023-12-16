class IirBeamFormer():

    filterbank = None


    def __init__(self, block_len, f_low, f_high, f_s, v_m):
        self._block_len = block_len
        self._f_low = int(2 * block_len/fs * pf_low)
        self._f_high = int(2 * block_len/fs * pf_high)
        self._fs = f_s
        self._vm = v_m

        pass

    def compute_filterbank(self, coord):
        n_phi = 360//1
        n_theta = 90//1
        theta_s = np.linspace(0, pi / 2, n_theta+1)
        phi_s = np.linspace(-pi, pi, n_phi+1)
        angles = np.array(np.meshgrid(-phi_s, pi/2 - theta_s)).T.reshape(-1,2)

        mat = R.from_euler("zy", angles, degrees=False)
        coord_t = (mat.as_matrix() @ coord.T).swapaxes(-1,-2).reshape(361,91,32,3)
        coord_t = coord_t.swapaxes(1,0)
        x = coord_t[:,:,:,0]
        y = coord_t[:,:,:,1]

        r_m = np.sqrt(x**2 + y**2)[... ,np.newaxis]
        phi_m = np.arctan2(y, x)[... ,np.newaxis]
        f = np.arange(self._block_len).reshape(1, 1, 1, -1) * self._fs / (2*self._block_len)

        t_s = sin(pi / 2 - phi_m) * r_m * (1 / self._v_m)
        fb = np.zeros((*np.squeeze(t_s).shape, self._block_len), dtype=np.complex64)
        fb[:,:,:,self._f_low:self._f_high] = exp(-1j * 2 * pi * f[:,:,:,self._f_low:self._f_high] * (t_s))

        self.filterbank = fb
        return fb

       pass

    def global_beam_sweep(self, signals):
        pass

    def local_beam_sweep(self, signals, center_direction):
        pass

    def listen_at(self, signals, direction):
        pass
