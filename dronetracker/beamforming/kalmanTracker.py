import numpy as np
from dataclasses import dataclass
from numpy import cos, sin, pi
import tomli
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata
import cv2 as cv
from math import radians
from datetime import datetime

import sys

sys.path.append("..")
from beamforming.tracker import Tracker
from beamforming.iirBeamformer import IirBeamFormer
from beamforming.kalman import KalmanFilter2D

from utils.mic_array import make_fancy_circ_array, calculate_umbrella_array
from utils.sphere import create_sphere
from utils.peakDetection import arg_max_detector, peak_detector2, arg_max_detector2


@dataclass
class Kalman_track_object:
    #     track_id: int
    track: np.ndarray
    k_filter: KalmanFilter2D
    color: str
    n_predictions: int = 0
    fname: str = ""


class KalmanTracker(Tracker):
    i = 0

    def __init__(self, config_file=None, **kwargs):
        print("Init Tracker")
        current_datetime = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.datetime_str = str(current_datetime)
        self.out_file = f'./out/{self.datetime_str}.csv'
        with open(self.out_file, 'w') as f:
            pass

        self.angle = None
        self.n_mics = None
        self.mic_order = None
        arr_param = None
        self.phi_m, self.r_m, self.coords = (None, None, None)

        self.angle_thresh = kwargs.get("angle_thresh", 2)


        self.block_len = kwargs.get("block_len", 2048)


        self.v_m = kwargs.get("v_m", 332)
#         self.beamformer = IirBeamFormer(self.block_len // 2, 500, 2000, 44100, 335)
        self.beamformer = IirBeamFormer(self.block_len // 2, v_m = self.v_m, **kwargs.get("beamformer_settings", None))

        self.sphere_size = kwargs.get("sphere_size", 1500)
        self.sphere_factor = kwargs.get("sphere_factor", 1.1) # how much more than the semisphere
        self.sphere = sphere = create_sphere(
            self.sphere_size, self.sphere_factor
        )  # Make spher sligthly bigger for peak detection
        self.phi = sphere["phi"]
        self.theta = sphere["theta"]

        self.max_blind_predict = kwargs.get("max_blind_predict", 10)
        self.tracker_threshold = kwargs.get("tracker_threshold", 1)

        r_projection = self.theta
        self.x_projection = cos(self.phi) * r_projection
        self.y_projection = sin(self.phi) * r_projection

        self.peak_detector_mask = self.make_peak_detector_mask()
        self.peak_det_settings = kwargs.get("peak_det_settings", None)


        self.alpha_gnss = kwargs.get("alpha_gnss", 0.9)
        self.c_lon = 8.8189
        self.c_lat = 47.22321


        self.save_tracks = kwargs.get("save_tracks", False)
        self.tracker_count = 0
        self.objects = []
        self.colors = [
            "#fc3fc4",
            "#aa67c9",
            "#d636ab",
            "#3c801f",
            "#0abf6a",
            "#c27f32",
            "#2ebac9",
            "#fc723f",
            "#33b039",
            "#9d42cf",
            "#155ca3",
            "#6e8209",
            "#47ffd1",
            "#72aab0",
            "#000000",
            "#fc3fc4",
            "#aa67c9",
            "#d636ab",
            "#3c801f",
            "#0abf6a",
            "#c27f32",
            "#2ebac9",
            "#fc723f",
            "#33b039",
            "#9d42cf",
            "#fc3fc4",
            "#aa67c9",
            "#d636ab",
            "#3c801f",
            "#0abf6a",
            "#c27f32",
            "#2ebac9",
            "#fc723f",
            "#33b039",
            "#9d42cf",
        ]


    def init_umbrella_array(self, angle=0, lat=None, lon=None):
        if lon is not None and lat is not None:
            self.c_lat = lat
            self.c_lon = lon
        self.n_mics = 32
        self.angle = angle
        self.mic_order = np.arange(self.n_mics)
        self.coords = calculate_umbrella_array(radians(angle), 0.01185 - 0.0016).T
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)

    def update_umbrella_array(self, angle):
        print("======= Update Umbrella Angle ======")
        self.coords = calculate_umbrella_array(radians(angle), 0.01185 - 0.0016).T
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)
        self.angle = angle
        print("====== Update Done ======")

    def needs_update(self, angle):
        return (np.abs(angle - self.angle) > self.angle_thresh)

    def update_pos(self, lat, lon):
        self.c_lat = self.c_lat * self.alpha_gnss + lat *(1-self.alpha_gnss)
        self.c_lon = self.c_lon * self.alpha_gnss + lon *(1-self.alpha_gnss)

    def init_config_array(self, config_file):
        with open(config_file, "rb") as f:
            config = tomli.load(f)
        self.n_mics = config["n_mic"]
        self.mic_order = config["mic_order"]
        arr_param = config.get("arr_param", None)
        if config.get("arr_type") == "umbrella":

            self.coords = calculate_umbrella_array(radians(arr_param["gamma"]), 0.01185 - 0.0016).T
            self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)
        else:
            self.phi_m, self.r_m, self.coords = make_fancy_circ_array(
                self.n_mics, **arr_param
            )

            self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)

    def get_sphere(self):
        return self.phi[: self.sphere_size], self.theta[: self.sphere_size]
#         return self.phi[: ], self.theta[: ]

    def make_peak_detector_mask(self):
        grid_x, grid_y = np.mgrid[-1.8:1.8:191j, -1.8:1.8:191j]
        mask = np.zeros_like(self.x_projection)
        mask[: int(self.sphere_size * np.mean([1, self.sphere_factor]))] = 1
        grid = griddata(
            (self.x_projection, self.y_projection),
            mask,
            (grid_x, grid_y),
            method="linear",
            fill_value=0.5,
        ).T
        mask = grid == 1
        return mask

    def _convert_into_cartesian(self, index):
        phi = self.phi[index]
        theta = self.theta[index]
        r = theta
        x = cos(phi) * r
        y = sin(phi) * r
        return np.array([x, y])

    def _create_kalman_object(self, pos_0):
        Ts = 0.01
        Qv = 200 # 500
        Q_pos = 300 # 200
        Q_vel = 600 # 300
        return KalmanFilter2D(Ts, Qv, Q_pos, Q_vel, pos_0[0], pos_0[1])

    def _make_blind_prediciton(self, tracking_object):
        new_pos = tracking_object.k_filter.run_blind_filter()
        tracking_object.track = np.vstack((tracking_object.track, new_pos[:2]))
        tracking_object.n_predictions += 1

    def _update_trackers(self, peaks, convert_to_cartesian=False):
        # If argmax peak detector, the index of the 1d response array 
        # must be converted into cartesian coordinates
        peaks_cartesian = peaks
        if convert_to_cartesian:
            peaks_cartesian = [self._convert_into_cartesian(peak) for peak in peaks]
        if len(self.objects) == 0:
            for peak in peaks_cartesian:
                kalman = self._create_kalman_object(peak)
                self.objects.append(
                    Kalman_track_object(
                        peak.reshape((-1, 2)), kalman, self.colors.pop(), fname=f'./out/{self.datetime_str}_{self.tracker_count}.txt'
                    )
                )
                self.tracker_count += 1
            return

        objects = []
        tracker_indices = [i for i in range(len(self.objects))]
        peak_indices = [i for i in range(len(peaks))]
        tracker_positions = [
            track_object.k_filter.make_prediction()[:2] for track_object in self.objects
        ]
        if len(peaks) > 0:
            cost_matrix = cdist(np.vstack(peaks_cartesian), np.vstack(tracker_positions))

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Iterate through nearest peaks from munkres
            for peak_ind, tracker_ind in zip(row_indices, col_indices):
                tracker_indices.remove(tracker_ind)
                peak_indices.remove(peak_ind)
                peak_pos = peaks_cartesian[peak_ind]
                tracker_pos = tracker_positions[tracker_ind]
                tracking_object = self.objects[tracker_ind]

                # is peak is to far away from assigned track, create new one, else continue tracking
                if np.linalg.norm(peak_pos - tracker_pos) < self.tracker_threshold:
                    new_pos = tracking_object.k_filter.run_filter(peak_pos)
                    tracking_object.track = np.vstack((tracking_object.track, new_pos[:2]))
                    tracking_object.n_predictions = 0
                else:
                    kalman = self._create_kalman_object(peak_pos)
                    objects.append(
                        Kalman_track_object(
                            peak_pos.reshape((-1, 2)), kalman, self.colors.pop(), fname=f'./out/{self.datetime_str}_{self.tracker_count}.txt'

                        )
                    )
                    self.tracker_count += 1

                    # if track ran too long on kalman without peak update, discard it
                    if tracking_object.n_predictions >= self.max_blind_predict:
                        self.colors.insert(0, tracking_object.color)
                        continue
                    self._make_blind_prediciton(tracking_object)
                objects.append(tracking_object)

        # Iterate through all existing trackers, that didnt match previously
        for tracker_ind in tracker_indices:
            tracking_object = self.objects[tracker_ind] # TODO Errors
            if tracking_object.n_predictions >= self.max_blind_predict:
                self.colors.insert(0, tracking_object.color)
                continue
            self._make_blind_prediciton(tracking_object)
            objects.append(tracking_object)

        # Create a new tracker for each peak, that wasn't used previously
        for peak_ind in peak_indices:
            peak_pos = peaks_cartesian[peak_ind]
            kalman = self._create_kalman_object(peak_pos)
            objects.append(
                Kalman_track_object(
                    peak_pos.reshape((-1, 2)), kalman, self.colors.pop(), fname=f'./out/{self.datetime_str}_{self.tracker_count}.txt'

                )
            )
            self.tracker_count += 1

        self.objects = objects

    def do_compass_correction(self, angle, peaks):
#         angle = 2*pi - angle
        R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        return [R @ peak for peak in peaks]

    def save_peaks(self, peaks):
        block_str = []
        for peak in peaks:
            track_phi = np.arctan2(peak[1],peak[0])
            track_theta = np.sqrt(peak[0]**2 + peak[1]**2)
            block_str.append(f'({track_phi:.5f}, {track_theta:.5f})')
        with open(self.out_file, 'a') as f:
            f.write(';'.join(block_str) + "\n")

    def write_trackers(self):
        for tracker in self.objects:
            if tracker.fname and tracker.track.shape[0] > 20:
                track_rad = np.zeros_like(tracker.track)
                with open(tracker.fname, 'w') as f:
                    f.write(np.array2string(np.arctan2(tracker.track[:,1], tracker.track[:,0])))
                    f.write("\n")
                    f.write(np.array2string(np.sqrt(tracker.track[:,0]**2 + tracker.track[:,1]**2)))



    def track(self, block, compass_angle=0):
        print("<><><><><><Tracker><><><><><><><><")
        block = block[:, self.mic_order]
        response = self.beamformer.global_beam_sweep(block)

        max_peak = arg_max_detector(response)
        max_val = response[max_peak[0]]

        grid_x, grid_y = np.mgrid[-1.8:1.8:191j, -1.8:1.8:191j]
        grid = griddata(
            (self.x_projection, self.y_projection),
            response,
            (grid_x, grid_y),
            method="linear",
            fill_value=0.5,
        ).T
        grid_cv = (grid/ np.max(grid) * 255).astype(np.uint8)
        peaks= peak_detector2(grid_cv, val_array=grid, area_mask=self.peak_detector_mask, sphere_factor=self.sphere_factor, **self.peak_det_settings)
#         peaks = arg_max_detector2(grid, min_height=10)
        peaks = self.do_compass_correction(compass_angle, peaks)
#         peaks = self.do_compass_correction(0, peaks)
        self.save_peaks(peaks)
        self._update_trackers(peaks)
        if self.save_tracks:
            self.write_trackers()
        # For debugging Purposes
        #         grid = (grid * 255).astype(np.uint8)
        #         cv.imwrite(f'./tmp/im{self.i}.png', grid)
        #         cv.imwrite(f'./tmp/hans{self.i}.png', grid)
        self.i += 1

        response = response / max(max_val, 20) # max Value
#         response = response /  # max Value
        print("<><><><><><Tracker Done><><><><><>")
        return response[: self.sphere_size], peaks, self.objects, max_val, None
#         return response[: ], peaks, self.objects, max_val, None
