"""Helper class for AsyncSamplesOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import threading

class APPOReplayBuffer(object):
    def __init__(self, maxsize = 200):
        self.lock = threading.Lock()
        self.maxsize= maxsize
        self.list = []
        return

    def put(self, element):
        self.lock.acquire()
        self.remove_if_full()
        self.list.append(element)
        self.lock.release()

    def get(self):  
        #No Polling, use condition object
        self.lock.acquire()
        if(len(self.list)==0):
            start = time.time()
            while(len(self.list)==0):
                self.lock.release()
                time.sleep(0.5)
                self.lock.acquire()

        element = self.list.pop()
        self.lock.release()
        return element

    def remove_if_full(self):
        if(len(self.list)==self.maxsize):
            self.list.pop(0)

    def qsize(self):
        return len(self.list)





