# --------------------------------------------------------------------------- #
#   Copyright (c) 2024-2026 Steve Kieffer                                     #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at                                   #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
# --------------------------------------------------------------------------- #

"""
Custom connection and cursor classes for logging purposes
"""

import logging
import os
import sqlite3
import time
import traceback
import uuid

OPEN_CURSORS = {}


def generate_new_uid(prefix=''):
    ts = time.time_ns()
    return f'{prefix}-{ts}-{uuid.uuid4()}'


class PlanLoggingConnection(sqlite3.Connection):
    """
    This connection class, and the associated cursor class, serve to log query plans,
    each time a query is made.
    """

    def cursor(self):
        sqlite_cursor = super().cursor()
        return PlanLoggingCursor(sqlite_cursor)


class PlanLoggingCursor:
    """
    Cursor formed by PlanLoggingConnection, to log query plans.
    """

    def __init__(self, sqlite_cursor):
        self.sqlite_cursor = sqlite_cursor
        self.logger = logging.getLogger(__name__)

    def execute(self, sql, params=None):
        params = params or []

        explain_sql = 'EXPLAIN QUERY PLAN ' + sql
        self.sqlite_cursor.execute(explain_sql, params)
        strat = self.sqlite_cursor.fetchall()

        self.logger.info(sql)
        self.logger.info(strat)

        return self.sqlite_cursor.execute(sql, params)

    def fetchone(self):
        return self.sqlite_cursor.fetchone()

    def fetchall(self):
        return self.sqlite_cursor.fetchall()

    def close(self):
        return self.sqlite_cursor.close()


class OpenCloseLoggingConnection(sqlite3.Connection):
    """
    This connection class, and the associated cursor class, serve to log opening and closing
    of connections and cursors. Can help to track down issues with multiple open cursors
    causing "database locked" errors.
    """

    def __init__(self, database, timeout=5.0, log_level=1):
        super().__init__(database, timeout=timeout)
        self.gremlite_uid = generate_new_uid(prefix='CONN')
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        note_pid = f'PID={os.getpid()}'
        self.log('~~~~~~ OP', extra=note_pid)

    def log(self, action, extra=''):
        if self.log_level < 2:
            return
        ts = time.time_ns()
        msg = f'{action} ({ts}) {self.gremlite_uid} {extra}'
        self.logger.info(msg)

    def cursor(self):
        sqlite_cursor = super().cursor()
        return OpenCloseLoggingCursor(self, sqlite_cursor, log_level=self.log_level)

    def commit(self) -> None:
        self.log('CMT')
        return super().commit()

    def close(self):
        self.log('~~~~~~ CL')
        return super().close()


class OpenCloseLoggingCursor:
    """
    Cursor formed by OpenCloseLoggingConnection, to log open/close events.

    log_level:
        1: Just record self (with stack trace) in the `OPEN_CURSORS` registry.
        2: Also record open/close events with INFO-level logging.
    """

    def __init__(self, connection, sqlite_cursor, log_level=1):
        self.gremlite_uid = generate_new_uid(prefix='CURS')
        self.log_level = log_level

        if self.log_level >= 1:
            OPEN_CURSORS[self.gremlite_uid] = traceback.format_stack()

        self.connection = connection
        self.sqlite_cursor = sqlite_cursor

        self.logger = logging.getLogger(__name__)
        self.log('OPN')

    def log(self, action, extra=''):
        if self.log_level < 2:
            return
        ts = time.time_ns()
        msg = f'{action} ({ts}) {self.gremlite_uid} {self.connection.gremlite_uid} {extra}'
        self.logger.info(msg)

    def execute(self, sql, params=None):
        params = params or []
        self.log('EXC', extra=sql)
        return self.sqlite_cursor.execute(sql, params)

    def fetchone(self):
        return self.sqlite_cursor.fetchone()

    def fetchall(self):
        return self.sqlite_cursor.fetchall()

    def close(self):
        result = self.sqlite_cursor.close()
        self.log('CLS')
        if self.log_level >= 1:
            del OPEN_CURSORS[self.gremlite_uid]
        return result


def print_open_cursor_traces(limit=5):
    """
    Convenience function to print a representation of all currently open cursors,
    when using an OpenCloseLoggingConnection.

    :param limit: upper limit on how many open cursors' full stack trace will be printed
    """
    N = len(OPEN_CURSORS)
    print(f'There are {N} open cursors.')
    for uid, trace in list(OPEN_CURSORS.items())[:min(N, limit)]:
        print("%" * 80)
        print(uid)
        for line in trace:
            print(line.rstrip('\n'))
