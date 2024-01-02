import threading
import time
import numpy as np

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from AudioInterface.buffer import RingBuffer
from AudioInterface.tcpReciever import TcpReciever
from AudioInterface.streamer import Streamer

class TcpStreamer(Streamer):

    def __init__(self, chanels=32):
        self._chanels = chanels
        self._buffer = RingBuffer(15, self._chanels)
        self._tcp_reciever = TcpReciever(self._buffer)

    def start_stream(self):
        self._tcp_reciever.start()

    def end_stream(self):
        self._tcp_reciever.stop()

    def get_block(self, block_size):
        return self._buffer.get_n(block_size)


if __name__ == "__main__":
    print('Hi')
    streamer = TcpStreamer()
    streamer.start_stream()
    time.sleep(20)
    streamer.end_stream()
