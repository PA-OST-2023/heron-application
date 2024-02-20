import socket
import threading
import time
import numpy as np
from buffer import RingBuffer

class TcpReceiver:
    def __init__(self, ip_addr, buffer, port=6666):
        self.host = ip_addr
        self.port = port
        self.tcp_conn_timeout = 5.0
        self.buffer = buffer
        self._running = False

    def start(self):
        streamer = threading.Thread(target=self._stream)
        self._running = True
        streamer.start()

    def stop(self):
        self._running = False
        pass

    def _stream(self):
        dataBuffer = bytearray()
        magicStartSequence = bytes("HERON666", "utf-8")

        headerSize = 20
        channelCount = 32
        blockSampleCount = 128
        audioBlockSize = channelCount * blockSampleCount * 2
        packetSize = audioBlockSize + headerSize

        lastPacketIndex = -1

        while self._running:  # Keep trying to connect
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect((self.host, self.port))
                    s.sendall(b"0")
                    print(f"Connected to {self.host}:{self.port}")
                    i = 0
                    lastReceived = time.time()

                    while self._running:  # Keep trying to receive data
                        try:
                            data = s.recv(1460 * 8)
                            lastReceived = time.time()
                            if not data:
                                continue
                            dataBuffer.extend(data)

                            headerIndex = dataBuffer.find(magicStartSequence)
                            if headerIndex != -1 and len(dataBuffer) >= packetSize:
                                packetIndex = int.from_bytes(
                                    dataBuffer[headerIndex + 8 : headerIndex + 12],
                                    "little",
                                )
                                timestamp = int.from_bytes(
                                    dataBuffer[
                                        headerIndex + 12 : headerIndex + headerSize
                                    ],
                                    "little",
                                )
                                audioData = dataBuffer[
                                    headerIndex
                                    + headerSize : headerIndex
                                    + headerSize
                                    + audioBlockSize
                                ]
                                dataBuffer = dataBuffer[headerIndex + packetSize :]
                                if len(audioData) != audioBlockSize:
                                    continue
                                decoded_data = np.frombuffer(
                                    audioData, dtype=np.int16
                                ).reshape(-1, 32)
                                self.buffer.append(decoded_data, decoded_data.shape[0])

                                if lastPacketIndex == -1:
                                    lastPacketIndex = packetIndex - 1
                                if packetIndex != lastPacketIndex + 1:
                                    print(
                                        f"Packet Drop Detected: {packetIndex}, Timestamp: {timestamp/10e9:.6f}"
                                    )
                                lastPacketIndex = packetIndex

                                i += 1
                                # if i % 100 == 0:
                                #     print(
                                #         f"Packet index: {packetIndex}, Timestamp: {timestamp/10e9:.6f}"
                                #     )

                        except (
                            socket.timeout,
                            TimeoutError,
                        ) as e:  # Don't care about short connection drops, as long as the connection is re-established
                            if time.time() - lastReceived > self.tcp_conn_timeout:
                                break
                            continue
                        except KeyboardInterrupt:
                            print("Interrupted by user")
                            raise KeyboardInterrupt
                        except OSError as e:
                            if e.strerror == "Stream closed":
                                print("Stream closed")
                                s.shutdown(socket.SHUT_RDWR)
                                break
                            else:
                                print("Trying to re-establish connection: ", e, type(e))
                                break

            except (socket.timeout, TimeoutError, ConnectionRefusedError) as e:
                pass
            except KeyboardInterrupt:
                print("Terminating")
                break


if __name__ == "__main__":
    ip = "192.168.33.80"
    buffer = RingBuffer(15, 32)
    reciever = TcpReceiver(ip, buffer)
    reciever.start()
    print("Start Streaming")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        reciever.stop()
        print("Done")
