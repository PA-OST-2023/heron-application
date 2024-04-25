import numpy as np
from numpy import cos, sin, pi
import scipy.io.wavfile as wavfile
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata
from math import radians, degrees

import sys

# from beamforming.tracker import Tracker
from iirBeamformer import IirBeamFormer

# from beamforming.kalman import KalmanFilter2D

from mic_array import make_fancy_circ_array, calculate_umbrella_array
from sphere import create_sphere
from peakDetection import arg_max_detector, peak_detector2, arg_max_detector2


class SoundDetector:

    def __init__(self, **kwargs):
        self.angle = None
        self.n_mics = None
        self.mic_order = None
        self.coords = None

        self.angle_thresh = kwargs.get("angle_thresh", 2)

        self.block_len = kwargs.get("block_len", 2048)

        self.v_m = kwargs.get("v_m", 332)
        #         self.beamformer = IirBeamFormer(self.block_len // 2, 500, 2000, 44100, 335)
        self.beamformer = IirBeamFormer(
            self.block_len // 2, v_m=self.v_m, **kwargs.get("beamformer_settings", None)
        )

        self.sphere_size = kwargs.get("sphere_size", 1500)
        self.sphere_factor = kwargs.get(
            "sphere_factor", 1.1
        )  # how much more than the semisphere
        self.sphere = sphere = create_sphere(
            self.sphere_size, self.sphere_factor
        )  # Make spher sligthly bigger for peak detection
        self.phi = sphere["phi"]
        self.theta = sphere["theta"]

        # Projections data used for the multi peak detector
        r_projection = self.theta
        self.x_projection = cos(self.phi) * r_projection
        self.y_projection = sin(self.phi) * r_projection

        self.peak_detector_mask = self.make_peak_detector_mask()
        self.peak_det_settings = kwargs.get("peak_det_settings", None)

    def init_umbrella_array(self, angle=0):
        """
        Initialize the umbrella Array


        Parameters
        ----------
        angle : float
            angel of the array arms
        """
        self.n_mics = 32
        self.angle = angle
        self.mic_order = np.arange(self.n_mics)
        self.coords = calculate_umbrella_array(angle, 0.01185 - 0.0016).T
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)

    def update_umbrella_array(self, angle):
        self.coords = calculate_umbrella_array(radians(angle), 0.01185 - 0.0016).T
        self.beamformer.compute_angled_filterbank(self.coords.T, self.phi, self.theta)
        self.angle = angle

    def needs_update(self, angle):
        return np.abs(angle - self.angle) > self.angle_thresh

    def get_sphere(self):
        """
        Generate sampling points for the steering vectors on a semisphere
        """
        return self.phi[: self.sphere_size], self.theta[: self.sphere_size]

    #         return self.phi[: ], self.theta[: ]

    def make_peak_detector_mask(self):
        """
        Create mask for the peakdetector to make it more stable on the edges

        """
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

    def do_compass_correction(self, angle, peaks):
        #         angle = 2*pi - angle
        R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        return [R @ peak for peak in peaks]

    def detectSounds(self, block, compass_angle=0):
        """
        Calculate the directions in spherical angles from where sounds come from
        returns the list uf 1d ndarrays containing phi and theta angle of a
        detected source
        if no sources were found the list is empty

        Parameters
        ----------
        block : ndarray
            2d array containing audioblock with `int16` type and shape (block_len, n_mics)
        compass_angle : float
            angle at of the array relative to north
        """

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
        grid_cv = (grid / np.max(grid) * 255).astype(np.uint8)
        peaks = peak_detector2(
            grid_cv,
            val_array=grid,
            area_mask=self.peak_detector_mask,
            sphere_factor=self.sphere_factor,
            **self.peak_det_settings,
        )

        # do correction to comapass angle, so that phi=0 means north. if
        # compass angle = 0 then, the return equals the input
        peaks = self.do_compass_correction(compass_angle, peaks)

        found_sources_directions = []
        for peak in peaks:
            found_sources_directions.append(np.array([np.arctan2(peak[1], peak[0]), np.sqrt(peak[0] ** 2 + peak[1] ** 2)]))

        response = response / max(max_val, 20)  # max Value
        #         response = response /  # max Value
        return found_sources_directions


#         return response[: ], peaks, self.objects, max_val, None

if __name__ == "__main__":
    block_len = 2048 *1
    settings = {
        "block_len": block_len,  # Lenght of the audio block used for the sound source detection
        "sphere_size": 2500,  # Number of samples on the Search space, which is a half sphere
        "sphere_factor": 1.2,  # Number of additional Samples on the half Sphere, 1 equals to a semisphere, 1.2 is a bit more. Required to be > 1 for the peak detector to work
        "angle_thresh": 2,  # min diffference for array angle to compute a new filterbank
        "v_m": 332,  # used sound speed
    }
    peak_detector_settings = {
        "max_height": 2**15,  # Max height for a peak to be detected as a peak
        "min_height": 5,  # Min Height for a peak to be detected as a peak
        "rel_max": 0.1,  # min relativce height of the lowest to the talles peak
    }
    # F_low and f_high set the frequency range in which the bemaformer operates.
    # Frequencies out of this range aren't considered in the beamformer
    beamformer_settings = {"f_low": 500, "f_high": 2000, "f_s": 44100}

    settings["beamformer_settings"] = beamformer_settings
    settings["peak_det_settings"] = peak_detector_settings

    # initialize Sound Detector
    soundDetector = SoundDetector(**settings)
    soundDetector.init_umbrella_array(angle=0) # set angle accordingly to the tilt angle of the array

    # read wav and foind sources every half second
    sr, data = wavfile.read('../data/2024-01-11--13-58-57.wav')
    soundDetector.update_umbrella_array(angle=45) # Angle of array in this file was 40 degrees
    for i in range(int(np.floor(data.shape[0]/sr))*2):
        found_sources = soundDetector.detectSounds(data[i*sr//2:i*sr//2+block_len])
        sources_in_degree = [angle/np.pi * 180 for angle in found_sources]
        [print(source) for source in sources_in_degree]
        print("-"*20 + f'{i}' + "-"*20)
