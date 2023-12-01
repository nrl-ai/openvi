import socket  # Import socket module
import os
import zipfile
import shutil

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object
# host = socket.gethostname() # Get local machine name
host = "0.0.0.0"
port = 12345  # Reserve a port for your service.
s.bind((host, port))  # Bind to the port
s.listen(5)  # Now wait for client connection.

zip_name = "socket_server.zip"
while True:
    c, addr = s.accept()  # Establish connection with client.
    with c:
        f = open(zip_name, "wb")
        print("Got connection from", addr)
        l = c.recv(1024)
        if l == b"":
            print("Receive test connection signal!")
            continue
        while l:
            print("Receiving...")
            f.write(l)
            l = c.recv(1024)
        f.close()
        print("Done Receiving")
        # c.send('Thank you for connecting')
        c.close()  # Close the connection

        # Handle the received .zip file
        base_path = os.path.abspath(__file__)
        directory_to_extract_to = os.path.join(
            base_path, "..", "..", "node_editor", "setting"
        )

        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        # shutil.unpack_archive(zip_name, directory_to_extract_to)
        print("Done extract received file!")
