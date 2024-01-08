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
        self.tracker = KalmanTracker(
            Path(__file__).parent / "configs" / "testfancy1.toml"
        )
        self.ui = UI(self.tracker, None)

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
