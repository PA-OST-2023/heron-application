import numpy as np
from time import sleep
import threading


class RingBuffer:
    def __init__(self, capacity, num_channels):
        self.capacity = np.uint32(2**capacity)
        self.ringMask = np.uint16(self.capacity - 1)
        self._data = np.empty((self.capacity, num_channels), dtype=np.int16)
        self.size = 0
        self.head = np.uint16(0)
        self.tail = np.uint16(0)
        self._available = 0
        #         self._semaphore = threading.Semaphore()
        #         self._lock = threading.Lock()
        #         self._isReadyEvent = threading.Event()
        #         self._readCondition = threading.Condition(self._lock)
        self._readCondition = threading.Condition()
        #         import ipdb; ipdb.set_trace()
        self._isReadReady = False

    def is_empty(self):
        return self.size == 0

    def get_available(self):
        #         return self.head - self.tail
        return self.size

    #         return self._available

    def is_full(self):
        return self.size == self.capacity

    def append(self, item, size):
        with self._readCondition:
            if self.is_full():
                self.tail = (self.tail + 1) & self.ringMask
            else:
                self.size += size
            self._data[self.head : self.head + size] = item
            self.head = (self.head + size) & self.ringMask
            self._readCondition.notify_all()

    #             print(self.get_available())
    #             print(f'{self.head = }')

    def get_all_available(self):
        return [self.data[(self.tail + i) % self.capacity] for i in range(self.size)]

    def get_n(self, num):
        with self._readCondition:
            if not self._readCondition.wait_for(
                lambda: self.get_available() >= num, timeout=10
            ):
                return None
            #             start = self.head - num
            indices = np.arange(self.head - num, self.head, 1)
            self.tail = self.head
            self.size = 0
            #             print('---------get------')
            #             print(f'{self.tail = }')
            #             return np.take(self._data, indices)
            return self._data[indices, :].copy()

    #         return [self.data[(self.tail + i) % self.capacity] for i in range(self.size)]

    def get_capacity(self):
        return self.capacity

    def get_size(self):
        return self.size


if __name__ == "__main__":

    def writer(buffer):
        for i in range(44100):
            data = np.array([i, i, i])
            buffer.append(data, 1)
            #         print('Data Writa')
            sleep(1 / 44100)

    def reader(buffer):
        while True:
            data = buffer.get_n(2048)
            if data is None:
                return
            print(f"{np.min(data)} --- { np.max(data)}")

    print("start")

    print("start")
    buffer = RingBuffer(15, 3)
    schribi = threading.Thread(target=writer, args=(buffer,))
    lesi = threading.Thread(target=reader, args=(buffer,))
    schribi.start()
    lesi.start()

    # Create a ring buffer with capacity of 5
#     buffer = RingBuffer(5)

# Add elements to the buffer
#     for i in range(7):
#         buffer.append(i)

# Get the elements currently stored in the buffer
#     print("Elements in the buffer:", buffer.get())

# Check buffer status
#     print("Is buffer empty?", buffer.is_empty())
#     print("Is buffer full?", buffer.is_full())
#     print("Buffer capacity:", buffer.get_capacity())
#     print("Buffer size:", buffer.get_size())
