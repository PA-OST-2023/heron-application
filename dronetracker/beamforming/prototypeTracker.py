import numpy as np
from numpy import cos, sin
import tomli

import sys

sys.path.append("..")
from beamforming.iirBeamformer import IirBeamFormer
from beamforming.tracker import Tracker
from utils.mic_array import make_fancy_circ_array

class ProtTracker(Tracker):

    i = 0
    def __init__(self, config_file):
        print('Init Tracker')
        with open(config_file, "rb") as f:
            config = tomli.load(f)
        self.arr_type = config['arr_type']
        self.n_mics = config['n_mic']
        self.mic_order = config['mic_order']
        arr_param = config.get('arr_param', None)
        self.phi, self.r = make_fancy_circ_array(self.n_mics, **arr_param)
#         self.phi = phi[self.mic_order]
#         self.r = r[self.mic_order]
        self.coords = np.vstack((cos(self.phi), sin(self.phi), np.zeros_like(self.phi))) * self.r
        self.beamformer = IirBeamFormer(1024, 500, 2000, 44100, 335)
        self.beamformer.compute_filterbank(self.coords.T)

        print('Tracker Initialized')

    def track(self, block):
        print('Do beamforming')
        print(block.shape)
        self.i+= 1
        block = block[:,self.mic_order]
        response = self.beamformer.global_beam_sweep(block)
        maxi = np.unravel_index(np.argmax(response, axis=None), response.shape)
        print('beamformed')
        print(f'{maxi}')
        print(f'=============={self.i}=============')
