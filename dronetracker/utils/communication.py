import socket
import json
import threading
import time

class Communication:
    def __init__(self, update_interval = 1.0):
        self.ip = None
        self.port = None
        self.running = False
        self.update_interval = update_interval
        self.data = None
        self.outgoingCommands = []

    def start(self, ip, port = 6667):
        self.ip = ip
        self.port = port
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def stop(self):
        self.running = False

    def getData(self):
        return self.data

    def sendCommand(self, command, value):
        self.outgoingCommands.append((command, value))

    def update(self):
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)  # Set a timeout
                    s.connect((self.ip, self.port))
                    try:
                        data = self.send_http_request(s, "/", "GET")         # Send GET request and print response
                        json_data = data.split('\r\n\r\n')[-1]
                        self.data = json.loads(json_data)
                        with open("./out/data.json", "w") as f:
                            f.write(json_data)
                        if(self.outgoingCommands):
                            outDict = {}
                            for command in self.outgoingCommands:
                                outDict[command[0]] = command[1]
                            json_data = json.dumps(outDict)
                            response = self.send_http_request(s, "/", "POST", json_data)

                    except OSError as e:
                        if(e.strerror == "Stream closed"):
                            print("Stream closed")
                            s.shutdown(socket.SHUT_RDWR)
                        else:
                            print("Trying to re-establish connection: ", e, type(e))
                        return None
                    except Exception as e:
                        print("Trying to re-establish connection: ", e, type(e))
                        return None

            except (socket.timeout, TimeoutError, ConnectionRefusedError) as e:
                pass
            except KeyboardInterrupt:
                print("Terminating")
            time.sleep(1 / self.update_interval)


    def send_http_request(self, s, path, method, data=None):
        request = f"{method} {path} HTTP/1.1\r\nHost: {s.getsockname()[0]}\r\n"     # Prepare the request
        if data:
            request += "Content-Type: application/json\r\n"
            request += f"Content-Length: {len(data)}\r\n"
        request += "\r\n"
        if data:
            request += data
        s.sendall(request.encode())             # Send the request
        response = s.recv(4096).decode()        # Receive the response
        return response


if __name__ == "__main__":
    com = Communication()
    com.start("192.168.33.80", 6667)

    t0 = time.time()
    while True:
        data = com.getData()
        if data:
            print(data)
#         print(com.getData())
        time.sleep(1)
        if(time.time() - t0 > 10):
            break
    com.stop()
