# --------------------------------------------------------------------------- #
#   Copyright (c) 2024 Steve Kieffer                                          #
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
This one is a study not of our library per se, but of sqlite itself.

EXPERIMENT
----------

Try building three tables assigning IDs to strings in sqlite, and test how fast they are on
searches for strings satisfying an order condition.

Table `strings0` uses no constraints, table `strings1` makes the ID an autoincrement primary key,
and `strings2` does that plus puts a uniqueness constraint on the strings as well.

We fill each table with the same set of random strings (we use UUID4's). Then we ask each
table to return the set of IDs of strings less than a given one (again a UUID4), and we see
how long each one takes.

RESULTS
-------

After running several times with the number of table rows being various powers of ten, from
10^3 to 10^6, the rule is: more constraints implies faster.

I observed quite a bit of variance from one run to the next as far as *how much* faster.
Sometimes the advantage seemed small; other times significant. What was pretty consistent
was that strings1 had only a small advantage over strings0, while strings2 could be twice as
fast as strings1. Here is a typical run, with 10^5 rows:

    strings0: 0.00601101
    strings1: 0.00523400
    strings2: 0.00207806

(times are in seconds).

Occasionally `strings0` could even be a little faster than `strings1`. For example, again
with 10^5 rows:

    strings0: 0.01352406
    strings1: 0.01441789
    strings2: 0.01072311

This makes sense, since I would only imagine that the uniqueness constraint on the strings
would require sqlite to build a reverse index, which would keep the strings sorted, and would
make "<" searches faster.
"""

import pathlib
import sqlite3
import tempfile
import time
import uuid


def main():
    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'
        con = sqlite3.Connection(path)
        cur = con.cursor()

        # strings0: no primary key, no unique
        cur.execute("CREATE TABLE strings0 (id INTEGER, v TEXT)")
        # strings1: add primary key
        cur.execute("CREATE TABLE strings1 (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
        # strings2: add both primary key and unique
        cur.execute("CREATE TABLE strings2 (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT UNIQUE)")

        # Rows per table:
        N = 10**5
        for k in range(N):
            s = str(uuid.uuid4())
            for i in range(3):
                if i == 0:
                    cur.execute(f"INSERT INTO strings{i} (id, v) VALUES (?, ?)", [k, s])
                else:
                    cur.execute(f"INSERT INTO strings{i} (v) VALUES (?)", [s])

        s = str(uuid.uuid4())
        for i in range(3):
            t0 = time.time()
            cur.execute(f"SELECT id FROM strings{i} WHERE v < ?", [s]).fetchall()
            t1 = time.time()
            print(f'strings{i}: {t1 - t0:.8f}')


if __name__ == "__main__":
    main()
