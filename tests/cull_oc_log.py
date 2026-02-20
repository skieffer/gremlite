# --------------------------------------------------------------------------- #
#   Copyright (c) 2026 Steve Kieffer                                          #
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
For culling open/close events out of the logs
"""

from enum import Enum
import pathlib
import re
import sys

TESTS_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent


class MemberRep(Enum):
    """
    Enum for different ways of representing the members of a co-occurence, when printing.
    """
    # Just say how many members there are, as a number:
    count = 'count'
    # Show number of members as in a histogram (i.e. by a bar of that length):
    histo = 'histo'
    # List all the members explicitly:
    list_all = 'list_all'


class CoOccurrence:
    """
    Represents a co-occurrence of entities.
    I.e. a set of things (cursors or connections) that were open at the same time,
    over some interval of time.
    """

    def __init__(self, members: dict, start_time):
        self.members = members.copy()
        self.start_time = start_time
        self.end_time = None

    def __str__(self):
        return self.to_string()

    def to_string(self, mr: MemberRep = MemberRep.count):
        if mr == MemberRep.list_all:
            members = [str(m) for m in self.members.values()]
        elif mr == MemberRep.histo:
            members = "*" * len(self.members)
        else:
            members = len(self.members)

        start = int(self.start_time[8:-3])

        if self.end_time is None:
            end = '-' * 8
            delta = 0
        else:
            end = int(self.end_time[8:-3])
            delta = end - start

        return f'({start}, {end}, {delta:8d}) {members}'

    def __len__(self):
        return len(self.members)

    def exit(self, uid, exit_time):
        """
        When a member exits, we can mark the end time of this co-occurrence,
        and return a new one, representing the members that remain.
        """
        self.end_time = exit_time
        new_members = {k: v for k, v in self.members.items() if k != uid}
        return CoOccurrence(new_members, exit_time)


class Connection:

    def __init__(self, uid, pid):
        self.uid = uid
        self.pid = pid

    def __str__(self):
        return f'CONN({self.pid})-{self.uid}'


class Cursor:

    def __init__(self, conn, uid):
        self.conn = conn
        self.uid = uid

    def __str__(self):
        return f'CURS-{self.uid}-CONN-{self.conn.uid}'


conn_op_cl_pattern = re.compile(r'~~~~~~ (\w\w) \((\d+)\) CONN-([-a-f0-9]+) PID=(\d+)')
conn_commit_pattern = re.compile(r'CMT \((\d+)\) CONN-([-a-f0-9]+)')
curs_pattern = re.compile(
    r'(\w\w\w) \((\d+)\) CURS-([-a-f0-9]+) CONN-([-a-f0-9]+) (.+$)'
)


class EventName:
    CONN_OP = 'OP'
    CONN_CL = 'CL'
    CONN_CMT = 'CMT'

    CURS_OPN = 'OPN'
    CURS_EXC = 'EXC'
    CURS_CLS = 'CLS'


def cull_oc_log(log_file_path='pytest.log'):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    conns = {}
    cursors = {}

    co_conns = []
    co_cursors = []

    for line in lines:
        # Is it a connection open/close event?
        if M := conn_op_cl_pattern.search(line):
            event, ts, uid, pid = M.groups()

            # Connection open:
            if event == EventName.CONN_OP:
                assert uid not in conns
                conn = Connection(uid, pid)
                conns[uid] = conn
                if len(conns) >= 2:
                    co_oc = CoOccurrence(conns, ts)
                    co_conns.append(co_oc)

            # Connection close:
            elif event == EventName.CONN_CL:
                assert uid in conns
                if len(conns) >= 2:
                    last_co_oc = co_conns[-1]
                    new_co_oc = last_co_oc.exit(uid, ts)
                    if len(new_co_oc) >= 2:
                        co_conns.append(new_co_oc)
                del conns[uid]

        # Is it a commit event?
        elif M := conn_commit_pattern.search(line):
            # Here maybe we want to note any open cursors, at the time
            # that the commit was attempted, and to which connections
            # those cursors belonged.
            ts, uid = M.groups()
            ...  # TODO

        # Is it a cursor event?
        elif M := curs_pattern.search(line):
            event, ts, uid, conn_uid, notes = M.groups()

            # Cursor open:
            if event == EventName.CURS_OPN:
                assert uid not in cursors
                assert conn_uid in conns
                conn = conns[conn_uid]
                curs = Cursor(conn, uid)
                cursors[uid] = curs
                if len(cursors) >= 2:
                    co_oc = CoOccurrence(cursors, ts)
                    co_cursors.append(co_oc)

            # Cursor close:
            elif event == EventName.CURS_CLS:
                assert uid in cursors
                if len(cursors) >= 2:
                    last_co_oc = co_cursors[-1]
                    new_co_oc = last_co_oc.exit(uid, ts)
                    if len(new_co_oc) >= 2:
                        co_cursors.append(new_co_oc)
                del cursors[uid]

            # Cursor execute:
            elif event == EventName.CURS_EXC:
                ...  # TODO

    return co_conns, co_cursors


def main(log_file_path='pytest.log'):
    co_conns, co_cursors = cull_oc_log(log_file_path)

    print()
    print("Connection co-occurrences:")
    for co in co_conns:
        print(co.to_string(mr=MemberRep.histo))

    print()
    print("Cursor co-occurrences:")
    for co in co_cursors:
        print(co.to_string(mr=MemberRep.histo))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = PROJECT_ROOT / 'pytest.log'

    main(path)
