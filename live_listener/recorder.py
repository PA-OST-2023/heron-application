import wave

class Recorder:
    def __init__(self, num_channels = 1):
        self._wavfile = None
        self._num_channels = num_channels
        self._is_recording = False

    def append(self, data):
        try:
            if self._is_recording:
                self._wavfile.writeframes(data.tobytes())
        except ValueError:
            print("Wav already closed")

    def start_recording(self, fname):
        self._wavfile = wave.open(fname, "wb")
        self._wavfile.setframerate(44100)
        self._wavfile.setnchannels(self._num_channels)
        self._wavfile.setsampwidth(2)
        self._is_recording = True

    def stop_recording(self):
        self._is_recording = False
        self._wavfile.close()


if __name__ == "__main__":
    import numpy as np

    recoder = Recorder(1)       # Mono

    freq = 500                  # Hz
    duration = 5                # seconds
    sample_rate = 44100         # Hz

    data = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * freq / sample_rate) * 32767).astype(np.int16)
    recoder.start_recording("test.wav")
    recoder.append(data)
    recoder.stop_recording()
