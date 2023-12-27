import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fft import fft, fftfreq, ifft

from beamforming.prototypeTracker import ProtTracker
from AudioInterface.waveStreamer import WavStreamer
from ui.liveBeamPlot import UI
from pathlib import Path

class Application ():

    fig = None
    ax = None
    ani = None

    def __init__(self):
        audio_file = Path(__file__).parent.parent / "data" / "random.wav"
#         audio_file = Path(__file__).parent.parent / "data" / "dyn.wav"
        self.streamer = WavStreamer(audio_file, 1024*4)
        self.block_len=1024*2
        self.tracker = ProtTracker(Path(__file__).parent / "configs" / "testfancy1.toml")
        self.ui = UI(self.tracker, self.streamer)



    def start(self):
        self.ui.run()
        return
        self.streamer.start_stream()
        i = 0
        while(True):
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print('----Done---')
                self.streamer.end_stream()
                break
            i += self.block_len
            self.tracker.track(block)


if __name__ == "__main__":
    print("Hello")
    print("Trello")
    app = Application()
    app.start()
