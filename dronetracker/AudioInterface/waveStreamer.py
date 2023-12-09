import pyaudio
import wave
import numpy as np
from time import sleep
import sys

sys.path.append("..")
from buffer import RingBuffer

from streamer import Streamer


class WavStreamer(Streamer):


    def __init__(self, audio_file, block_size):
        self.__audio_file = audio_file

        self.__block_size = block_size

        self.__wf = wave.open(self.__audio_file)
        self.__channels = self.__wf.getnchannels()
        self.__sr = self.__wf.getframerate()
        self.__buffer = RingBuffer(15, self.__channels)

        self.__stream = None
        self.__p = None



    def start_stream(self):
        self.__p = pyaudio.PyAudio()
        self.__stream = self.__p.open(format=self.__p.get_format_from_width(self.__wf.getsampwidth()),
                channels=self.__channels,
                rate=self.__sr,
                output=True,
                frames_per_buffer=self.__block_size,  # Set the block size here
                stream_callback=self.__callback())
        self.__stream.start_stream()

    def end_stream(self):
        self.__stream.stop_stream()
        self.__stream.close()
        self.__wf.close()

        # Terminate the PyAudio object
        self.__p.terminate()

    def __load_buffer(self, data):
        print('load')
        self.__buffer.append(data, data.shape[0])


    def get_block(self):
        print('---------READ--------')
        return self.__buffer.get_n(self.__block_size)

    def __callback(self):
        def callback(in_data, frame_count, time_info, status):
            data = self.__wf.readframes(frame_count)
            data_np = np.frombuffer(data, dtype=np.int16)
            data_shaped = data_np.reshape(-1, self.__channels)
            self.__load_buffer(data_shaped)
            return (data, pyaudio.paContinue)
        return callback


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    audio_file = '../../data/dyn2.wav'
#     audio_file = '../../data/dyn.wav'
    streamer = WavStreamer(audio_file, 1024*4)
    streamer.start_stream()
    data = []
    while(True):
        block = streamer.get_block()
        if block is None:
            print('----Done---')
            break
        data.append(block)
        print(f'read {block.shape}')
    streamer.end_stream()
    mic_signals = np.vstack(data)
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(mic_signals[:, 0].flatten())
    ax[1].plot(mic_signals[:, 1].flatten())
    plt.show()
