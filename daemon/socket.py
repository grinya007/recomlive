from collections import namedtuple
import os, socket


Message = namedtuple('Message', 'data, response')

class Socket(object):
    def __init__(self, host, port, dispatch, max_size = 1024):
        self.host = host
        self.port = port
        self.max_size = max_size
        self.dispatch = dispatch

    def listen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        while True:
            data, address = self.sock.recvfrom(self.max_size)
            self.dispatch(Message(data, lambda rdata: self.sock.sendto(rdata, address)))

