import numpy as np
import threading


class RingBuffer:
    def __init__(self, capacity, num_channels):
        self.capacity = int16(2**capacity -1)
        self.data = np.empty_like((self.capacity, num_channels), dtype=np.int16)
        self.size = 0
        self.head = 0
        self.tail = 0
        self.available = 0

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def append(self, item):
        if self.is_full():
            self.tail = (self.tail + 1) % self.capacity
        else:
            self.size += 1
        self.data[self.head] = item
        self.head = (self.head + 1) & self.capacity

    def get(self):
        return [self.data[(self.tail + i) % self.capacity] for i in range(self.size)]

    def get_capacity(self):
        return self.capacity

    def get_size(self):
        return self.size

if __name__ == "__name__":
    # Create a ring buffer with capacity of 5
    buffer = RingBuffer(5)

    # Add elements to the buffer
    for i in range(7):
        buffer.append(i)

    # Get the elements currently stored in the buffer
    print("Elements in the buffer:", buffer.get())

    # Check buffer status
    print("Is buffer empty?", buffer.is_empty())
    print("Is buffer full?", buffer.is_full())
    print("Buffer capacity:", buffer.get_capacity())
    print("Buffer size:", buffer.get_size())
