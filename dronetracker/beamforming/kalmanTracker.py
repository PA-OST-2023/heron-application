import numpy as np
from dataclasses import dataclass
from numpy import cos, sin
import tomli
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata

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
    color: str
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

        self.max_blind_predict = 10

        self.r = self.theta
        self.x = cos(self.phi) * self.r
        self.y = sin(self.phi) * self.r

        self.objects = []
        self.colors = ["#fc3fc4",
                        "#2ebac9",
                        "#fc723f",
                        "#33b039",
                        "#9d42cf",
                        "#155ca3",
                        "#6e8209",
                        "#47ffd1"]

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

    def _make_blind_prediciton(self, tracking_object):
        new_pos = tracking_object.k_filter.run_blind_filter()
        tracking_object.track = np.vstack((tracking_object.track, new_pos[:2]))
        tracking_object.n_predictions += 1

    def _update_trackers(self, peaks, threshold = 0.3):
        peaks_cartesian = [self._convert_into_cartesian(peak) for peak in peaks]
        if len(self.objects) == 0:
            for peak in peaks_cartesian:
                kalman = self._create_kalman_object(peak)
                self.objects.append(Kalman_track_object(
                                    peak.reshape((-1,2)),
                                    kalman,
                                    self.colors.pop())
                                    )
            return

        objects = []
        tracker_indices = [i for i in range(len(self.objects))]
        peak_indices = [i for i in range(len(peaks))]
        tracker_positions = [track_object.k_filter.make_prediction()[:2] for track_object in self.objects]
        cost_matrix = cdist(np.vstack(peaks_cartesian), np.vstack(tracker_positions))

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for peak_ind, tracker_ind in zip(row_indices, col_indices):
            tracker_indices.remove(tracker_ind)
            peak_indices.remove(peak_ind)
            peak_pos = peaks_cartesian[peak_ind]
            tracker_pos = tracker_positions[tracker_ind]
            tracking_object = self.objects[tracker_ind]
            if np.linalg.norm(peak_pos - tracker_pos) < threshold:
                new_pos = tracking_object.k_filter.run_filter(peak_pos)
                tracking_object.track = np.vstack((tracking_object.track, new_pos[:2]))
                tracking_object.n_predictions = 0
            else:

                kalman = self._create_kalman_object(peak_pos)
                objects.append(Kalman_track_object(peak_pos.reshape((-1,2)), kalman,self.colors.pop()))

                if tracking_object.n_predictions >= self.max_blind_predict:
                    self.colors.insert(0, tracking_object.color)
                    continue
                self._make_blind_prediciton(tracking_object)
            objects.append(tracking_object)

        for tracker_ind in tracker_indices:
            tracking_object = self.objects[tracker_ind]
            if tracking_object.n_predictions >= self.max_blind_predict:
                self.colors.insert(0, tracking_object.color)
                continue
            self._make_blind_prediciton(tracking_object)
            objects.append(tracking_object)

        for peak_ind in peak_indices:
            peak_pos = peaks_cartesian[peak_ind]
            kalman = self._create_kalman_object(peak_pos)
            objects.append(Kalman_track_object(peak_pos.reshape((-1,2)), kalman,self.colors.pop()))

        self.objects = objects
        return
        peak = peaks_cartesian[0]
        predition = self.objects[0].k_filter.make_prediction()[:2]
        if np.linalg.norm(peak-predition) < 0.3:
            new_pos = self.objects[0].k_filter.run_filter(peak, 0)
            self.objects[0].track = np.vstack((self.objects[0].track, new_pos[:2]))
        else:
            self.colors.insert(0, self.objects[0].color)
            kalman = self._create_kalman_object(peak)
            self.objects[0] = Kalman_track_object(peak.reshape((-1,2)), kalman,self.colors.pop())
        return



    def track(self, block):
        print("===========Tracker==========")
        block = block[:, self.mic_order]
        response = self.beamformer.global_beam_sweep(block)


        peaks = arg_max_detector(response)
        self._update_trackers(peaks)
        max_val = response[peaks[0]]
        response = response / response[peaks[0]]

        grid_x, grid_y = np.mgrid[-1.6:1.6:150j, -1.6:1.6:150j]
        grid = griddata(
            (self.x, self.y), response, (grid_x, grid_y), method="linear"
        ).T

        print('<><><><><><><><><>')
        print(self.objects[0].track[-1])
        return response, peaks, self.objects, max_val, None

