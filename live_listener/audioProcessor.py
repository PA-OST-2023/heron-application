from buffer import RingBuffer
from tcpReceiver import TcpReceiver
from recorder import Recorder
from beamformer import Beamformer

import sounddevice as sd
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import convolve


class AudioProcessor:
    def __init__(self,ip_addr, port=6666, channels=32, fs=44100, block_len=128):
        self._stream_active = False
        self._fs = fs
        self._block_len = block_len
        self._name = ip_addr
        self._channels = channels
        self._buffer = RingBuffer(20, self._channels)
        self._recorder = Recorder(1)                    # Mono output
        self._tcp_reciever = TcpReceiver(ip_addr, self._buffer)

        self.stream = sd.OutputStream(samplerate=self._fs, channels=1, callback=self.callback, blocksize=self._block_len, dtype=np.int16)
        self.beamformer = Beamformer(self._channels, self._fs)

        # FFT Approach
        self._block_len_fft = self._block_len * 2
        self._last_block_fft = np.zeros(self._block_len, dtype=np.complex64)    # Mono output

        # Delay-Line Approach
        self._delay_line = np.zeros((channels, block_len), dtype=np.float32)

        self.update_delays(self.beamformer.calculate_delays(0, 0))   # Initialize delays (dummy)
        
        

    def start_stream(self):
        if self._stream_active:
            return
        self._tcp_reciever.start()
        self.stream.start()
        self._buffer.clear()
        self._stream_active = True

    def end_stream(self):
        if not self._stream_active:
            return
        self._stream_active = False
        self._tcp_reciever.stop()
        self.stream.stop()

    def start_recording(self, fname):
        if not self._stream_active:
            return
        self._recorder.start_recording(fname)

    def stop_recording(self):
        if not self._stream_active:
            return
        self._recorder.stop_recording()


    def update_delays(self, delays):
        self._delays = delays

        # FFT Approach
        f = (np.arange(self._block_len_fft).reshape(1, -1) * self._fs / self._block_len_fft)
        self._delay_filter = np.exp(-1j * 2 * np.pi * f * self._delays)

        # Delay-Line Approach
        self._delay_line_taps = 128
        n = np.arange(self._delay_line_taps)
        h = np.sinc(n - delays * self._fs).astype(np.float64)      # Impulse response of the delay line
        
        # window = np.hamming(self._delay_line_taps)      # Windowing function to reduce side lobes
        # h *= window
        h /= np.sum(h)              # Normalize filter to ensure unity gain
        self._delay_line_weights = h
        
    

    # Internal functions
    def callback(self, outdata, frames, time, status):
        data = self._buffer.get_n(frames)
        if data is None:
            return
        data = data.astype(np.float32) / 32767.0            # Use float64 for processing
        mono = self.process_beamformer_delay_line(data)
        self._recorder.append(mono)

        mono = self.process_filter(mono)
        mono = self.process_compressor(mono)

        outdata[:] = (mono * 32767.0).astype(np.int16).reshape(-1, 1)      # Convert back to int16 for output
        if(self._buffer.get_size() > 5000):
            print("Buffer has been cleared to reduce latency")
            self._buffer.clear()


    def process_beamformer_delay_line(self, input):
        output = np.zeros(self._block_len, dtype=np.float32)
        for i in range(self._block_len):
            self._delay_line = np.roll(self._delay_line, 1, axis=1)     # Shift the delay line for one sample to the right (for each channel)
            self._delay_line[:,0] = input[i]                            # Update the first row of the delay line with the input
            product = self._delay_line * self._delay_line_weights       # Performe element wise multiplication for delay line and weights
            output[i] = np.sum(product) / self._channels                # Sum the products across the channels
        return output


    def process_beamformer_fft(self, input):
        spectrum = fft(input.T, n=self._block_len_fft, axis=-1)      
        output = ifft(self._delay_filter * spectrum, n=self._block_len_fft)
        sum_output = np.sum(output, axis=0) / self._channels
        sum_overlapped = sum_output[:self._block_len] + self._last_block_fft
        self._last_block_fft = sum_output[self._block_len:]
        return sum_overlapped.real
    
    


    def process_filter(self, input):        # Mono Low Pass Filter
        return input
    
    def process_compressor(self, input):    # Mono Compressor
        return input



if __name__ == "__main__":
    import time

    ip = "192.168.33.80"
    proc = AudioProcessor(ip)
    proc.start_stream()
    print("Start Streaming")

    proc.start_recording("out.wav")
    try:
        time.sleep(100)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        proc.stop_recording()
        proc.end_stream()
        print("Done")