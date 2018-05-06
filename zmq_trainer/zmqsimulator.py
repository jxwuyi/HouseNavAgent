#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: zmqsimulator.py

import multiprocessing as mp
import threading
import atexit
import os, sys
from abc import abstractmethod, ABCMeta
from six.moves import queue
import weakref
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

def dumps(obj):
    return msgpack.dumps(obj, use_bin_type=True)

def loads(buf):
    return msgpack.loads(buf)

def ensure_proc_terminate(proc):
    if isinstance(proc, list):
        for p in proc:
            ensure_proc_terminate(p)
        return
    def stop_proc_by_weak_ref(ref):
        proc = ref()
        if proc is None:
            return
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join()
    assert isinstance(proc, mp.Process)
    atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))

class SimulatorProcess(mp.Process):
    def __init__(self, idx, pipe_c2s, pipe_s2c, config=None):
        super(SimulatorProcess, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

        self.config = config

    @abstractmethod
    def _build_player(self):
        pass

    def run(self):
        try:
            player = self._build_player()
            assert player is not None
        except Exception as e:
            print('[ERROR] <ZMQSimulator> Fail to create player for <{}>, Msg = {}'.format(self.identity, e), file=sys.stderr)
            return
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        # s2c_socket.set_hwm(5)
        s2c_socket.connect(self.s2c)

        state = player.current_state()
        reward, isOver = 0, False
        while True:
            c2s_socket.send(dumps(
                (self.identity, state, reward, isOver)),
                copy=False)
            action = loads(s2c_socket.recv(copy=False).bytes)
            reward, isOver = player.action(action)
            state = player.current_state()


class SimulatorMaster(object):
    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)


    def send_loop(self):
        while True:
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)

    def recv_loop(self):
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident, state, reward, isOver = msg
                self.recv_message(ident, state, reward, isOver)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    def __del__(self):
        self.context.destroy(linger=0)

    @abstractmethod
    def recv_message(self, ident, state, reward, isOver):
        """
        Do something about the agent named "ident" after getting its output.
        """
        pass

    def send_message(self, ident, action):
        """
        Send action to the agent named "ident".
        """
        self.send_queue.put([ident, dumps(action)])
