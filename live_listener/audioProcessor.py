from buffer import RingBuffer
from tcpReceiver import TcpReceiver
from recorder import Recorder
import sounddevice as sd
import numpy as np

class AudioProcessor:
    def __init__(self,ip_addr, port=6666, chanels=32):
        self.name = ip_addr
        self._chanels = chanels
        self._buffer = RingBuffer(20, self._chanels)
        self._recorder = Recorder(1)    # Mono
        self._tcp_reciever = TcpReceiver(ip_addr, self._buffer)

        self.fs = 44100
        self.stream = sd.OutputStream(samplerate=self.fs, channels=1, callback=self.callback, blocksize=64, dtype=np.int16)

    def start_stream(self):
        self._tcp_reciever.start()
        self.stream.start()
        self._buffer.clear()

    def end_stream(self):
        self._tcp_reciever.stop()
        self.stream.stop()

    def get_block(self, block_size):
        return self._buffer.get_n(block_size)

    def start_recording(self, fname):
        self._recorder.start_recording(fname)

    def stop_recording(self):
        self._recorder.stop_recording()

    def callback(self, outdata, frames, time, status):
        data = self.get_block(frames)
        if data is None:
            return
        output = self.process(data)
        self._recorder.append(output)
        outdata[:] = output

        if(self._buffer.get_size() > 5000):
            print("Buffer has been cleared to reduce latency")
            self._buffer.clear()


    def process(self, input):
        left = input[:, 3]    # Extract channel 3 as left channel
        return left.reshape(-1, 1)
        # return np.hstack((left.reshape(-1, 1), right.reshape(-1, 1)))


if __name__ == "__main__":
    import time

    ip = "192.168.33.80"
    proc = AudioProcessor(ip)
    proc.start_stream()
    print("Start Streaming")

    proc.start_recording("out.wav")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        proc.stop_recording()
        proc.end_stream()
        print("Done")
