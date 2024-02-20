from buffer import RingBuffer
from tcpReceiver import TcpReceiver

class TcpStreamer:
    def __init__(self,ip_addr, port=6666, chanels=32):
        self.name = ip_addr
        self._chanels = chanels
        self._buffer = RingBuffer(15, self._chanels)
        self._tcp_reciever = TcpReceiver(ip_addr, self._buffer)

    def start_stream(self):
        self._tcp_reciever.start()

    def end_stream(self):
        self._tcp_reciever.stop()

    def get_block(self, block_size):
        return self._buffer.get_n(block_size)

    def start_recording(self, fname):
        self._buffer.start_recording(fname)

    def stop_recording(self):
        self._buffer.stop_recording()


if __name__ == "__main__":
    import time

    ip = "192.168.33.80"
    streamer = TcpStreamer(ip)
    streamer.start_stream()
    print("Start Streaming")

    streamer.start_recording("out.wav")
    try:
        time.sleep(100)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        streamer.stop_recording()
        streamer.end_stream()
        print("Done")
