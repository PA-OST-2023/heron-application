from tcpStreamer import TcpStreamer
from communication import Communication
from datetime import datetime
from pathlib import Path
import json
import time

class Recorder:
    def __init__(self, ip):
        self.ip = ip
        self.comm = Communication()
        self.streamer = TcpStreamer(self.ip)
        if not (Path(__file__).parent / "out").exists():    # check if out folder exists, if not create it
            (Path(__file__).parent / "out").mkdir()
        
    def record(self, live=False):
        self.comm.start(self.ip)
        self.streamer.start_stream()
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wav_path = Path(__file__).parent / "out" / f"{file_name}.wav"
        self.streamer.start_recording(str(wav_path))
        print(f"Start Recording: {str(wav_path)}")
        print("Press Ctrl+C to stop recording")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            with open(Path(__file__).parent / "out" / f"{file_name}.json", "w") as f:
                json.dump(self.comm.getData(), f)
            self.comm.stop()
            self.streamer.stop_recording()
            self.streamer.end_stream()
            print("Done")


if __name__ == "__main__":
    ip = "192.168.33.80"
    rec = Recorder(ip)
    rec.record()
