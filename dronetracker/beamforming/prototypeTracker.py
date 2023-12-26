import numpy as np
from numpy import cos, sin
import tomli
import plotly.graph_objects as go
import time

import sys

sys.path.append("..")
from beamforming.iirBeamformer import IirBeamFormer
from beamforming.tracker import Tracker
from beamforming.kalman import KalmanFilter2D
from utils.mic_array import make_fancy_circ_array
from utils.sphere import create_sphere

class ProtTracker(Tracker):

    i = 0
    fig = None
    def __init__(self, config_file):
        print('Init Tracker')
        with open(config_file, "rb") as f:
            config = tomli.load(f)
        self.arr_type = config['arr_type']
        self.n_mics = config['n_mic']
        self.mic_order = config['mic_order']
        arr_param = config.get('arr_param', None)
        self.phi, self.r = make_fancy_circ_array(self.n_mics, **arr_param)
        self.coords = np.vstack((cos(self.phi), sin(self.phi), np.zeros_like(self.phi))) * self.r

        # Precompute Filterbanks
        self.beamformer = IirBeamFormer(1024, 500, 2000, 44100, 335)
        self.sphere = sphere = create_sphere(1500)
        self.phi = sphere['theta']
        self.theta = sphere['phi']
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)

        Ts = 0.01
        Qv = 500
        Q_pos = 200
        Q_vel = 100
        self.kalman = KalmanFilter2D(Ts, Qv, Q_pos, Q_vel)


#         self.beamformer.compute_filterbank(self.coords.T)
#         self.beamformer.beamsearch(self.r, self.phi)

        print('Tracker Initialized')


    def _convert_into_cartesian(self, index):
        phi = self.phi[index]
        theta = self.theta[index]
        r = theta
        x = cos(phi)*r
        y = sin(phi)*r
        return np.array([x,y])

    def track(self, block):
        print('Do beamforming')
        print(block.shape)
        self.i+= 1
        block = block[:,self.mic_order]
        response = self.beamformer.global_beam_sweep(block)
#         self.fig.data[0].z = response
        time.sleep(0.01)
        maxi = np.unravel_index(np.argmax(response, axis=None), response.shape)
        pos = self._convert_into_cartesian(maxi)
        self.kalman.run_filter(pos, 0)
        pos_es = self.kalman.get_position()
        response = response /response[maxi]
        print('beamformed')
        print(f'========================================{maxi}')
        print(f'=============={self.i}=============')
        return response,pos, pos_es, self.phi, self.theta
