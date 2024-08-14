from buffer import RingBuffer
from tcpReceiver import TcpReceiver
from recorder import Recorder
from beamformer import Beamformer

import sounddevice as sd
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import convolve, firwin, lfilter
from numba import jit
import time


class AudioProcessor:
    def __init__(self,ip_addr, port=6666, channels=32, fs=44100, block_len=512):
        self._stream_active = False
        self._stream_stopping = False
        self._fs = fs
        self._block_len = block_len
        self._name = ip_addr
        self._channels = channels
        self._buffer = RingBuffer(20, self._channels)
        self._recorder = Recorder(1)                    # Mono output
        self._tcp_reciever = TcpReceiver(ip_addr, self._buffer, port)

        self.stream = sd.OutputStream(samplerate=self._fs, channels=1, callback=self.callback, blocksize=self._block_len, dtype=np.int16)
        self.beamformer = Beamformer(self._channels, self._fs)

        self._max_buffer_size = 25000
        self._initial_buffer_clear = True

        # FFT Approach
        self._block_len_fft = self._block_len * 2
        self._last_block_fft = np.zeros(self._block_len, dtype=np.complex64)    # Mono output

        # Delay-Line Approach
        self._delay_line_taps = 80
        self._delay_line_offset = 20
        self._delay_line = np.zeros((self._channels, self._delay_line_taps), dtype=np.float32)

        # Band-Pass Filter
        self._filter_taps = 48
        self._filter_buffer = np.zeros(self._filter_taps - 1)

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
        self._stream_stopping = True
        try:
            while self._stream_active:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
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
        n = np.arange(self._delay_line_taps)
        h = np.sinc(n - (delays * self._fs) - self._delay_line_offset).astype(np.float32)      # Impulse response of the delay line
        self._delay_line_weights = h
        
    
    # Internal functions
    def callback(self, outdata, frames, t, status):
        data = self._buffer.get_n(frames)
        if data is None or not self._stream_active:
            data = np.zeros((frames, 1), dtype=np.int16)

        if self._initial_buffer_clear:
            self._buffer.clear()
            self._initial_buffer_clear = False
            print("Initial buffer cleared")

        data = data.astype(np.float32) / 32767.0            # Use float32 for processing
        mono = self.process_beamformer_delay_line(data)
        self._recorder.append(mono)

        mono = self.process_filter(mono, 1400, 1900)
        mono = self.process_compressor(mono, -10, 8, 40)

        outdata[:] = (mono * 32767.0).astype(np.int16).reshape(-1, 1)      # Convert back to int16 for output
        if(self._buffer.get_size() > self._max_buffer_size):
            self._buffer.clear()
            print("Buffer has been cleared to reduce latency")

        if self._stream_stopping:
            self._stream_active = False
            self._stream_stopping = False


    def process_beamformer_delay_line(self, input):
        return delay_line(self._delay_line, self._delay_line_weights, input)


    def process_beamformer_fft(self, input):
        spectrum = fft(input.T, n=self._block_len_fft, axis=-1)
        output = ifft(self._delay_filter * spectrum, n=self._block_len_fft)
        sum_output = np.sum(output, axis=0) / self._channels
        sum_overlapped = sum_output[:self._block_len] + self._last_block_fft
        self._last_block_fft = sum_output[self._block_len:]
        return sum_overlapped.real


    def process_filter(self, input, f_min, f_max):        # Mono Low Pass Filter
        output = np.zeros(self._block_len, dtype=np.float32)
        h = firwin(self._filter_taps, [f_min, f_max], pass_zero=False, fs=self._fs)
        input = np.concatenate((self._filter_buffer, input))
        convolved = lfilter(h, 1.0, input)
        self._filter_buffer = input[-(self._filter_taps - 1):]
        valid_start = self._filter_taps - 1
        valid_end = valid_start + self._block_len
        output[:] = convolved[valid_start:valid_end]
        return output
    

    def process_compressor(self, input, threshold=-20, ratio=2, make_up_gain=0):
        ratio = min(ratio, 0.01)
        threshold_lin = 10**(threshold / 20)
        rms = np.sqrt(np.mean(input**2))            # Calculate the input signal's envelope (RMS)
        if rms > threshold_lin:                     # Calculate gain reduction factor
            gain_reduction = (rms / threshold_lin)**(1 - 1/ratio)
        else:
            gain_reduction = 1.0
        output = input * gain_reduction * 10**(make_up_gain / 20)    # Apply gain reduction
        return output


@jit
def delay_line(delay_line_object, weights, input):
    output = np.zeros(input.shape[0], dtype=np.float32)
    for i in range(input.shape[0]):
        delay_line_object[:, 1:] = delay_line_object[:, :-1]   # This is faster than np.roll
        delay_line_object[:,0] = input[i]                            # Update the first row of the delay line with the input
        output[i] = np.sum(delay_line_object * weights) / delay_line_object.shape[0]
    return output


if __name__ == "__main__":
    import time

    ip = "192.168.40.80"
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
