import re
import socket

from collections import namedtuple
Word = namedtuple("Word", ["word", "lemma", "pos", "animate"])
pos_re = re.compile('([A-Z]+)\((.*?)\)')


class Frog(object):
    def __init__(self, port):
        self.BUFSIZE = 4096
        self.port = port

    def tag(self, text):
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.settimeout(120.0)
        self.socket.connect(('localhost', self.port))

        text = text.strip(' \t\n')
        text = text.encode('utf-8') + b'\r\nEOT\r\n'
        self.socket.sendall(text)
        done = False
        output = []
        while not done:
            data = b""
            while not data or data[-1] != b'\n':
                more = self.socket.recv(self.BUFSIZE)
                if not more: break
                data += more
            for line in data.strip(' \t\r\n').split('\n'):
                line = line.strip()
                if line == 'READY':
                    done = True
                    break
                elif line:
                    index, word, lemma, _, pos, _ = line.split('\t')
                    if index == '1':
                        output.append([])
                    pos = pos_re.search(pos)
                    head = pos.group(1)
                    output[-1].append(Word(word, lemma, head, 'inanimate'))
        self.socket.close()
        return output
