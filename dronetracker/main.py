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
        self.xf = fftfreq(self.block_len, 1/44100)[:self.block_len//2] # TODO SAMPLINGRATE
        self.tracker = ProtTracker('./configs/testfancy1.toml')


    def plotCall(self):
        def animate(i):
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print('----Done---')
                self.streamer.end_stream()
                return
            chanel = block[:,0]
            chanel2 = block[:,1]
            sf = fft(chanel, self.block_len * 2)
            sf2 = fft(chanel2, self.block_len * 2)
            phat = (sf * sf2)/np.abs(sf * sf2)
            steeze = ifft(phat) * 22500
            self.ax.clear()
#             self.ax.set_ylim(0.0,100)
#             self.ax.set_xlim(0.0,1000)
            self.ax.grid()
#             self.ax.semilogx(self.xf[0:self.block_len//8], 2/self.block_len * np.abs(sf[0:self.block_len//8]))
            self.ax.plot(np.abs(steeze)[self.block_len -100: self.block_len + 100])

        return animate

    def lego(self):
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
#             chanel = block[:,0]
# #             chanel2 = block[:,1]
#             sfs = fft(block.T, self.block_len * 2)
#             print(sfs.shape)
#             print(i / 44100)
#        self.fig, self.ax = plt.subplots() 
#        self.ani = FuncAnimation(plt.gcf(), self.plotCall(), 100)
#        self.ax.set_ylim(0,200)
#         self.ax.grid(True)
#        plt.show()


if __name__ == "__main__":
    print("Hello")
    app = Application()
    app.lego()
