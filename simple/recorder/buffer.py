import numpy as np
import wave
import threading

class RingBuffer:
    def __init__(self, capacity, num_channels):
        self.capacity = np.uint32(2**capacity)
        self.ringMask = np.uint32(self.capacity - 1)
        self._data = np.empty((self.capacity, num_channels), dtype=np.int16)
        self.size = 0
        self.head = np.uint32(0)
        self.tail = np.uint32(0)
        self._available = 0
        self._readCondition = threading.Condition()
        self._isReadReady = False
        self._write_to_wav = False
        self._wavfile = None

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
        with self._readCondition:
            if self.is_full():
                self.tail = (self.tail + 1) & self.ringMask
            else:
                self.size += size
            self._data[self.head : self.head + size] = item
            self.head = (self.head + size) & self.ringMask
            self._readCondition.notify_all()

            if self._write_to_wav:
                try:
                    self._wavfile.writeframes(item.tobytes())
                except ValueError:
                    print("Wav already closed")

    def get_all_available(self):
        return [self.data[(self.tail + i) % self.capacity] for i in range(self.size)]

    def get_n_from_head(self, num):           # get n first samples from top
        with self._readCondition:
            if not self._readCondition.wait_for(
                lambda: self.get_available() >= num, timeout=10
            ):
                return None
            indices = np.arange(self.head - num, self.head, 1)
            self.size = 0       # reset size since we discard all data behind tail
            return self._data[indices, :].copy()
        
    def get_n(self, num):           # get n last samples in buffer
        with self._readCondition:
            if not self._readCondition.wait_for(
                lambda: self.get_available() >= num, timeout=10
            ):
                return None
            indices = np.arange(self.tail, self.tail + num, 1)
            self.tail = (self.tail + num) & self.ringMask
            self.size -= num
            return self._data[indices, :].copy()

    def get_capacity(self):
        return self.capacity

    def get_size(self):
        return self.size

    def start_recording(self, fname):
        self._wavfile = wave.open(fname, "wb")
        self._wavfile.setframerate(44100)
        self._wavfile.setnchannels(32)
        self._wavfile.setsampwidth(2)
        self._write_to_wav = True

    def stop_recording(self):
        self._write_to_wav = False
        self._wavfile.close()


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
