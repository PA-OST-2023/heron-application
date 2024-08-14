import numpy as np

class RingBuffer:
    def __init__(self, capacity, num_channels):
        self.capacity = np.uint32(2**capacity)
        self.ringMask = np.uint32(self.capacity - 1)
        self._data = np.empty((self.capacity, num_channels), dtype=np.int16)
        self.size = 0
        self.head = np.uint32(0)
        self.tail = np.uint32(0)
        self._available = 0
        self._isReadReady = False

    def is_empty(self):
        return self.size == 0

    def get_available(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity
    
    def clear(self):
        self.size = 0
        self.head = 0
        self.tail = 0

    def append(self, item, size):
        if self.is_full():
            self.tail = (self.tail + 1) & self.ringMask
        else:
            self.size += size
        self._data[self.head : self.head + size] = item
        self.head = (self.head + size) & self.ringMask

    def get_all_available(self):
        return [self.data[(self.tail + i) % self.capacity] for i in range(self.size)]

    def get_n_from_head(self, num):           # get n first samples from top
        if self.get_available() < num:
            return None
        indices = np.arange(self.head - num, self.head, 1)
        self.size = 0       # reset size since we discard all data behind tail
        return self._data[indices, :].copy()
        
    def get_n(self, num):           # get n last samples in buffer
        if self.get_available() < num:
            return None
        indices = np.arange(self.tail, self.tail + num, 1)
        self.tail = (self.tail + num) & self.ringMask
        self.size -= num
        return self._data[indices, :].copy()

    def get_capacity(self):
        return self.capacity

    def get_size(self):
        return self.size


if __name__ == "__main__":
    from time import sleep

    def writer(buffer):
        for i in range(44100):
            data = np.array([i, i, i])
            buffer.append(data, 1)
            sleep(1 / 44100)

    def reader(buffer):
        while True:
            data = buffer.get_n(2048)
            if data is None:
                return
            print(f"{np.min(data)} --- { np.max(data)}")

    print("start")

    buffer = RingBuffer(15, 3)
    schribi = threading.Thread(target=writer, args=(buffer,))
    lesi = threading.Thread(target=reader, args=(buffer,))
    schribi.start()
    lesi.start()

    sleep(3)
    schribi.join()
    lesi.join()
    print("end")
