import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fft import fft, fftfreq, ifft

from beamforming.prototypeTracker import ProtTracker
from AudioInterface.waveStreamer import WavStreamer

class Application ():

    fig = None
    ax = None
    ani = None

    def __init__(self):
        audio_file = '../data/dyn.wav'
        self.streamer = WavStreamer(audio_file, 1024*1)
        self.block_len=1024*2
        self.tracker = ProtTracker('./configs/testfancy1.toml')


    def start(self):
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
    app = Application()
    app.start()
