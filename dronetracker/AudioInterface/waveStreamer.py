import pyaudio
import wave
import numpy as np

# Define the callback function
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    data_np = np.frombuffer(data, dtype=np.int16)
    import ipdb; ipdb.set_trace()
    return (data, pyaudio.paContinue)

# Open the wave file
wf = wave.open('../../data/dyn.wav', 'rb')

# Create an instance of PyAudio
p = pyaudio.PyAudio()

# Set the desired block size
block_size = 2048  # for example, 2048 frames

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
while stream.is_active():
    pass  # Here you can do other tasks if needed

# Stop and close the stream
stream.stop_stream()
stream.close()
wf.close()

# Terminate the PyAudio object
p.terminate()
