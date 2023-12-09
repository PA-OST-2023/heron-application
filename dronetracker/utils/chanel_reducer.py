import wave
import numpy as np


with wave.open('../../data/dyn.wav', 'rb') as wf:
    n_frames = wf.getnframes()
    frames = wf.readframes(n_frames)
data_np = np.frombuffer(frames, dtype=np.int16)
data_shaped = data_np.reshape(-1, 32)
data_trunc = data_shaped[:,(0,15)]


with wave.open('../../data/dyn2.wav', 'wb') as wf:
    wf.setframerate(44100)
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.writeframes(data_trunc.tobytes())

