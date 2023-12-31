import numpy as np
from dataclasses import dataclass
from numpy import cos, sin
import tomli

import sys

sys.path.append("..")
from beamforming.tracker import Tracker
from beamforming.iirBeamformer import IirBeamFormer
from beamforming.kalman import KalmanFilter2D

from utils.mic_array import make_fancy_circ_array
from utils.sphere import create_sphere
from utils.peakDetection import arg_max_detector




@dataclass
class Kalman_track_object:
#     track_id: int
    track: np.ndarray
    k_filter: KalmanFilter2D
    n_predictions: int=0

class KalmanTracker(Tracker):
    def __init__(self, config_file=None):
        print("Init Tracker")

        with open(config_file, "rb") as f:
            config = tomli.load(f)
        self.n_mics = config["n_mic"]
        self.mic_order = config["mic_order"]
        arr_param = config.get("arr_param", None)
        self.phi_m, self.r_m, self.coords = make_fancy_circ_array(self.n_mics, **arr_param)

        self.beamformer = IirBeamFormer(1024, 500, 2000, 44100, 335)
        self.sphere = sphere = create_sphere(1500)
        self.phi = sphere["theta"]
        self.theta = sphere["phi"]
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)


        self.objects = []

    def _convert_into_cartesian(self, index):
        phi = self.phi[index]
        theta = self.theta[index]
        r = theta
        x = cos(phi) * r
        y = sin(phi) * r
        return np.array([x, y])

    def _create_kalman_object(self, pos_0):
        Ts = 0.01
        Qv = 500
        Q_pos = 200
        Q_vel = 100
        return KalmanFilter2D(Ts, Qv, Q_pos, Q_vel, pos_0[0], pos_0[1])

    def _update_trackers(self, peaks):
        peaks_cartesian = [self._convert_into_cartesian(peak) for peak in peaks]
        if len(self.objects) == 0:
            for peak in peaks_cartesian:
                kalman = self._create_kalman_object(peak)
                self.objects.append(Kalman_track_object(peak.reshape((-1,2)), kalman))
            return
        peak = peaks_cartesian[0]
        predition = self.objects[0].k_filter.make_prediction()[:2]
        if np.linalg.norm(peak-predition) < 0.2:
            new_pos = self.objects[0].k_filter.run_filter(peak, 0)
            self.objects[0].track = np.vstack((self.objects[0].track, new_pos[:2]))
        else:
            kalman = self._create_kalman_object(peak)
            self.objects[0] = Kalman_track_object(peak.reshape((-1,2)), kalman)
        return



    def track(self, block):
        print("===========Tracker==========")
        block = block[:, self.mic_order]
        response = self.beamformer.global_beam_sweep(block)

        peaks = arg_max_detector(response)
        self._update_trackers(peaks)

        response = response / response[peaks[0]]
        print('<><><><><><><><><>')
        print(self.objects[0].track[-1])
        return response, peaks, self.objects[0].track, None

