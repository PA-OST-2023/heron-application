from audioProcessor import AudioProcessor
from communication import Communication
import time


class Main():
    def __init__(self):
        self.ip = "192.168.33.80"

        self.sample_rate = 44100
        self.block_len = 128
        self.mic_num = 32

        self.com = Communication()
        self.proc = AudioProcessor(self.ip, 6666, self.mic_num, self.sample_rate, self.block_len)

        self.running = False

    def __del__(self):
        self.stop()
    
    def start(self):
        self.running = True

        self.com.start(self.ip, 6667)
        if(self.com.getData() == None):     # Wait for connection
            time.sleep(0.5)
        
        self.run()
        self.stop()


    def run(self):
        theta = 90
        phi = 0

        arm_angle = self.com.getData()["sensor_angle"]
        self.proc.beamformer.update_arm_angle(arm_angle)
        delays = self.proc.beamformer.calculate_delays(phi, theta)
        self.proc.update_delays(delays)

        self.proc.start_stream()

        while self.running:
            time.sleep(1)

    def stop(self):
        self.proc.end_stream()
        self.com.stop()
        

main = Main()
main.start()


