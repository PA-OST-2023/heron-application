import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fft import fft, fftfreq, ifft

from beamforming.prototypeTracker import ProtTracker
from beamforming.kalmanTracker import KalmanTracker
from AudioInterface.waveStreamer import WavStreamer
from AudioInterface.tcpStreamer import TcpStreamer
from ui.liveBeamPlot import UI
from pathlib import Path


class Application:
    fig = None
    ax = None
    ani = None
    tracker = None

    def __init__(self):
        audio_file = Path(__file__).parent.parent / "data" / "random.wav"
        #         audio_file = Path(__file__).parent.parent / "data" / "dyn.wav"
#         self.streamer = self._setup_wav_streamer()
        self.block_len = 1024 * 2
        beamformer_settings = {"f_low": 500,
                                "f_high": 2000,
                                "f_s": 44100}
        peak_detector_settings = {"min_height": 10,
                        "max_height": 2**15,
                        "rel_max": 0.2}
        streamer_settings = {}
        tracker_settings = {"block_len": self.block_len,
                "sphere_size": 1500,
                "sphere_factor": 1.1,
                "max_blinf_predict": 10,
                "angle_thresh": 2,
                "alpha_gnss": 0.9}
        tracker_settings['beamformer_settings'] = beamformer_settings
        tracker_settings['peak_det_settings'] = peak_detector_settings
        json_port = 6667
        self.ui = UI(streamer_settings, tracker_settings, json_port=json_port, use_compass=True
                )

    def _init_tracker(self):
        pass

    def _setup_tcp_streamer():
        return TcpStreamer()

    def _setup_wav_streamer(self):
        audio_file = Path(__file__).parent.parent / "data" / "random.wav"
        return WavStreamer(audio_file, 1024 * 4)

    def start(self):
        self.ui.run()
        return
        self.streamer.start_stream()
        i = 0
        while True:
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print("----Done---")
                self.streamer.end_stream()
                break
            i += self.block_len
            self.tracker.track(block)


if __name__ == "__main__":
    print("Hello")
    print("Trello")
    app = Application()
    app.start()
