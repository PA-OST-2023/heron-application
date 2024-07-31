import numpy as np
import wave

class Recorder:
    def __init__(self, num_channels = 1, sample_rate = 44100):
        self._wavfile = None
        self._num_channels = num_channels
        self._sample_rate = sample_rate
        self._is_recording = False

    def append(self, data):
        try:
            if self._is_recording:
                data = (data * 32767).astype(np.int16)
                self._wavfile.writeframes(data.tobytes())
        except ValueError:
            print("Wav already closed")

    def start_recording(self, fname):
        self._wavfile = wave.open(fname, "wb")
        self._wavfile.setframerate(self._sample_rate)
        self._wavfile.setnchannels(self._num_channels)
        self._wavfile.setsampwidth(2)
        self._is_recording = True

    def stop_recording(self):
        self._is_recording = False
        self._wavfile.close()


if __name__ == "__main__":
    freq_0 = 500                # Hz
    freq_1 = 1000               # Hz
    duration = 5                # seconds
    sample_rate = 44100         # Hz

    data_0 = np.sin(2 * np.pi * np.arange(sample_rate * duration) * freq_0 / sample_rate)
    data_1 = np.sin(2 * np.pi * np.arange(sample_rate * duration) * freq_1 / sample_rate)

    recoder = Recorder(1, sample_rate)      # Mono
    recoder.start_recording("mono.wav")
    recoder.append(data_0)
    recoder.stop_recording()

    recoder = Recorder(2, sample_rate)      # Stereo
    recoder.start_recording("stereo.wav")
    recoder.append(np.array([data_0, data_1]).T)
    recoder.stop_recording()
    