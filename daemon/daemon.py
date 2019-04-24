import os, signal, time

class Daemon(object):
    def __init__(self, pidfile, onstart):
        self.pidfile = pidfile
        self.onstart = onstart

    def command(self, cmd):
        method_name = '_' + str(cmd)
        method = getattr(self, method_name, self._unknown_command)
        return method()

    def _start(self):
        if self._is_running():
            raise Exception('Already running')
        pid = os.fork()
        if pid:
            os.waitpid(pid, 0)
            os._exit(0)
        os.umask(0)
        os.setsid()
        pid = os.fork()
        if pid:
            os._exit(0)
        self._write_pidfile()
        self.onstart()

    def _stop(self):
        pid = self._read_pidfile()
        if pid != None and self._is_running(pid):
            os.kill(pid, signal.SIGTERM)
            while self._is_running(pid):
                time.sleep(0.1)
        else:
            raise Exception('Not running')

    def _restart(self):
        try:
            self._stop()
        except Exception:
            pass
        finally:
            self._start()

    def _is_running(self, pid = None):
        running = False
        if pid == None:
            pid = self._read_pidfile()
        if pid != None and os.path.isdir('/proc/{}'.format(pid)):
            running = True
        return running

    def _unknown_command(self):
        raise ValueError('Unknown command called')

    def _read_pidfile(self):
        pid = None
        if os.path.isfile(self.pidfile):
            fh = open(self.pidfile)
            spid = fh.read().strip()
            if spid.isdigit():
                pid = int(spid)
        return pid

    def _write_pidfile(self):
        open(self.pidfile, 'w').write(str(os.getpid()))

