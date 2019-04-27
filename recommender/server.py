from .recommender import Recommender
from daemon import Daemon, Socket
from threading import Thread
from queue import Queue

import os, time

def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class Server():
    def __init__(
            self,
            host,
            port,
            logfile,
            pidfile,
            queue_limit,
            documents_n,
            persons_n,
            recs_limit,
            invalidate_after):
        self.host               = host
        self.port               = port
        self.logfile            = logfile
        self.pidfile            = pidfile
        self.queue_limit        = queue_limit
        self.documents_n        = documents_n
        self.persons_n          = persons_n
        self.recs_limit         = recs_limit
        self.invalidate_after   = invalidate_after
        self.chkdir(logfile)
        self.chkdir(pidfile)
        self.logfile_fh         = open(logfile, 'a')

    @lazyprop
    def daemon(self):
        return Daemon(self.pidfile, self.onstart)

    @lazyprop
    def socket(self):
        return Socket(self.host, self.port, self.dispatch)

    @lazyprop
    def queue(self):
        return Queue()

    @lazyprop
    def recommender(self):
        return Recommender(
                self.documents_n,
                self.persons_n,
                self.invalidate_after,
                self.recs_limit)

    def chkdir(self, sfile):
        sdir = os.path.dirname(sfile)
        if not os.path.isdir(sdir):
            os.makedirs(sdir)

    def command(self, cmd):
        return self.daemon.command(cmd)

    def onstart(self):
        self.thread = Thread(target=self.dispatcher)
        self.thread.start()
        self.socket.listen()
        self.recommender # init

    def dispatch(self, message):
        if self.queue.qsize() >= self.queue_limit:
            message.response(self._pack_response('QFULL'))
        else:
            self.queue.put(message)

    def dispatcher(self):
        while True:
            message = self.queue.get()
            try:
                method, did, pid = message.data.decode('ascii').split(',')
                if method == 'RECR':
                    self.recommender.record(did, pid)
                elif method == 'RECM':
                    recs = self.recommender.recommend(did, pid)
                    message.response(self._pack_response('OK', recs))
                elif method == 'RR':
                    self.recommender.record(did, pid)
                    recs = self.recommender.recommend(did, pid)
                    message.response(self._pack_response('OK', recs))
                elif method == 'PH':
                    dids = self.recommender.person_history(pid)
                    message.response(self._pack_response('OK', dids))
                else:
                    raise Exception
            except:
                message.response(self._pack_response('BADMSG'))

            self.queue.task_done()

    def _pack_response(self, status, data = []):
        return bytes(','.join([status] + data), 'ascii')



