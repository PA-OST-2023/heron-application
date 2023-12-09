import pyaudio
import wave
import numpy as np
from time import sleep

sys.path.append("..")
from buffer import Buffer

from streamer import Streamer


class WavStreamer(Streamer):

    self.__stream = None

    def __init__(self, audio_file, block_size):
        self.__audio_file = audio_file
        self.__wf = wave.open(self.__audio_file)
        self.__p = pyaudio.PyAudio

        self.__block_size = block_size

        self.__channels = self.__wf.getnchannels()
        self.__sr = self.__wf.getframerate()



    def start_stream(self):
        self.__stream = self.__p.open(format=p.get_format_from_width(self.__wf.getsampwidth()),
                channels=self.__channels,
                rate=self.__sr,
                output=True,
                frames_per_buffer=block_size,  # Set the block size here
                stream_callback=self.__callback())


        pass

    def __load_buffer(self, data):
        pass

    def __callback(self):
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            data_np = np.frombuffer(data, dtype=np.int16)
            self.__load_buffer(data_np)
            return (data, pyaudio.paContinue)
        return callback



def die():
    print('die')
    stream.stop_stream()
    stream.close()
    wf.close()

    # Terminate the PyAudio object
    p.terminate()
# Define the callback function
def callback(in_data, frame_count, time_info, status):
    global i
    data = wf.readframes(frame_count)
    data_np = np.frombuffer(data, dtype=np.int16)
    print('shit')
    print(data_np.shape)
    print(status)
    print(i)
    i+= 1
    return (data, pyaudio.paContinue)

# Open the wave file
wf = wave.open('../../data/dyn2.wav', 'rb')
i = 0
# Create an instance of PyAudio
p = pyaudio.PyAudio()

# Set the desired block size
block_size = 2048 * 4  # for example, 2048 frames

# Open a stream with the wave file's parameters and the specified block size
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=block_size,  # Set the block size here
                stream_callback=callback)

# Start the stream
stream.start_stream()

# Wait for stream to finish
# while stream.is_active():
#     print('------')
#     sleep(0.1)
#     pass  # Here you can do other tasks if needed
sleep(40)

# Stop and close the stream
stream.stop_stream()
stream.close()
wf.close()

# Terminate the PyAudio object
p.terminate()
