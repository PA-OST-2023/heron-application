from audioProcessor import AudioProcessor
from communication import Communication
from xboxcontroller import XboxController
import numpy as np
import threading
import time


class Main():
    def __init__(self):
        self.ip = "192.168.40.80"

        self.sample_rate = 44100
        self.block_len = 512
        self.mic_num = 32

        self.com = Communication()
        self.proc = AudioProcessor(self.ip, 6666, self.mic_num, self.sample_rate, self.block_len)

        self.controller_connected = False

        self.theta = 0
        self.phi = 0
        self.arm = 0
        self.running = False

    def __del__(self):
        self.stop()

    def start(self):
        self.running = True
        threading.Thread(target=self.controller_thread).start()

        self.com.start(self.ip, 6667)
        if(self.com.getData() == None):     # Wait for connection
            time.sleep(0.5)
        self.run()

    def stop(self):
        self.running = False
        self.proc.end_stream()
        self.com.stop()

    def run(self):
        self.proc.start_stream()
        try:
            while self.running:
                json = self.com.getData()
                if json != None:
                    self.arm = json["sensor_angle"]
                    self.proc.beamformer.update_arm_angle(self.arm)

                print(f"Theta: {self.theta:.1f}, Phi: {self.phi:.1f}, Arm: {self.arm:.1f}")
                delays = self.proc.beamformer.calculate_delays(self.phi, self.theta)
                self.proc.update_delays(delays)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Terminating")
        finally:
            self.stop()


    def controller_thread(self):
        self.controller = XboxController()
        while self.running:
            stick = self.controller.read()
            if stick != (-1, -1):
                self.controller_connected = True
                self.theta = np.sqrt(stick[0]**2 + stick[1]**2) * 90
                self.phi = np.arctan2(stick[0], stick[1]) * 180 / np.pi + 180
                self.theta = np.clip(self.theta, 0, 90)
                self.phi = np.clip(self.phi, 0, 360)
            else:
                self.controller_connected = False
            time.sleep(0.1)
        

main = Main()
main.start()


