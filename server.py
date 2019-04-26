#!/usr/bin/env python3
from recommender import Server
import os, sys

Server(
        os.getenv('RECOM_SERVER_HOST',                  'localhost'),
        int(os.getenv('RECOM_SERVER_PORT',              5000)),
        os.getenv('RECOM_SERVER_LOGFILE',               'var/log/server.log'),
        os.getenv('RECOM_SERVER_PIDFILE',               'var/run/server.pid'),
        int(os.getenv('RECOM_SERVER_QLIMIT',            10000)),
        int(os.getenv('RECOM_SERVER_DOCLIMIT',          2000)),
        int(os.getenv('RECOM_SERVER_PERSONLIMIT',       2000)),
        int(os.getenv('RECOM_SERVER_RECSLIMIT',         5)),
        int(os.getenv('RECOM_SERVER_INVALIDATEAFTER',   5)),
        ).command(sys.argv[1])

