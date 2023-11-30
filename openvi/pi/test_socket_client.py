import socket  # Import socket module
import sys, os
import argparse


def check_conn(host="127.0.0.1", port=12345):
    logs = []
    sock = socket.socket()
    result = sock.connect_ex((host, port))
    if result == 0:
        print("Port is open")
        logs.append("Port is open")
        sock.close()
        return True, logs
    else:
        print("Port is not open")
        logs.append("Port is not open")
        sock.close()
        return False, logs


def socket_client(host, port, zip_path):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # host = socket.gethostname() # Get local machine name
    # host = '172.20.10.3
    # port = 12345                 # Reserve a port for your service.
    logs = []

    s.connect((host, port))
    f = open(zip_path, "rb")
    print("Sending...")
    logs.append("Sending ...")
    l = f.read(1024)
    while l:
        print("Sending...")
        logs.append("Sending ...")
        s.send(l)
        l = f.read(1024)
    f.close()
    print("Done Sending")
    logs.append("Done Sending")

    s.close  # Close the socket when done
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip_path",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "socket_client.zip")
        ),
    )

    parser.add_argument("--ip_addr", type=str, default="127.0.0.1")

    parser.add_argument("--port", type=str, default="12345")

    args = parser.parse_args()
    zip_path = args.zip_path
    host = args.ip_addr
    port = args.port

    socket_client(host, port, zip_path)
