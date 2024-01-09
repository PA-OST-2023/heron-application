import socket
import json
import time

def send_http_request(s, path, method, data=None):
    # Prepare the request
    request = f"{method} {path} HTTP/1.1\r\nHost: {s.getsockname()[0]}\r\n"
    if data:
        request += "Content-Type: application/json\r\n"
        request += f"Content-Length: {len(data)}\r\n"
    request += "\r\n"
    if data:
        request += data

    # Send the request
    s.sendall(request.encode())

    # Receive the response
    response = s.recv(4096).decode()
    return response

def read_json(ip):
    ip = "192.168.33.80"
    port = 6667
    path = "/"
    index = 0

    once = True

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)  # Set a timeout
            s.connect((ip, port))

            try:
                # Send GET request and print response
                response = send_http_request(s, path, "GET")
                print(f"GET Response: {response}")

                # Send POST request with JSON data
                json_data = json.dumps({"index": index})
                response = send_http_request(s, path, "POST", json_data)
                print(f"POST Response: {response}")

                return response

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
    return None

    return
if __name__ == "__main__":
    main()
