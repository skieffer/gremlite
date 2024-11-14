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
Experiment
==========

Our goal here is to do some stress testing on filtering with `has()` steps.
We want to compare the efficiency of different filtering strategies.

We build a graph in which each vertex has a "first name" and "last name".
We will use UUID4's for the values, and we will record these as properties
"first" and "last".

Our graph will have no edges, since they are irrelevant for this experiment.

For each "last," we will have 5 to 15 vertices (number chosen at random in this
range) with this value as last name, and having random values for "first".

For our queries, we will locate all the vertices with a given "last", and then
filter down to those among these whose "first" lies in the "first half of the
alphabet," which means that it's < '8', since these are UUID4's.

Results
=======

We have just two filtering "strategies" to test: "plain", where all filtering is
performed "on the query side" i.e. within SQLite queries, and the "heuristic", where
some filtering is performed on the query side, and some "on the element side" i.e.
applying tests to actual property values in Python.

The results show that the queries were about 40 times faster when using the heuristic.

To be precise, queries took an average of about 17 to 19 ms without the heuristic,
but only about 0.4 to 0.5 ms with.

An example run:

    Will generate 2000 "last names."
    Graph will have about 20000 vertices.

    Building graph...
    Build took 2.404s

    Used filtering heuristic: False
    Made 10 queries.
    Average query time: 0.017012s

    Used filtering heuristic: True
    Made 10 queries.
    Average query time: 0.000469s
"""

import pathlib
import random
import tempfile
import time
import uuid

from gremlin_python.process.traversal import P, TextP
from gremlin_python.process.anonymous_traversal import traversal

from gremlite import get_g, GremliteConfig, SQLiteConnection


def main():
    # Number of last names:
    N = 2000
    print()
    print(f'Will generate {N} "last names."')
    print(f'Graph will have about {10*N} vertices.')
    # Number of queries to make:
    Q = 10

    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'

        remote = SQLiteConnection(path)
        g = traversal().with_remote(remote)

        last_names = []

        # Build the graph
        print()
        print(f'Building graph...')
        t0 = time.time()
        for i in range(N):
            last_name = str(uuid.uuid4())
            last_names.append(last_name)
            for j in range(random.randint(5, 15)):
                first_name = str(uuid.uuid4())
                g.add_v('person') \
                    .property('last', last_name).property('first', first_name) \
                    .iterate()
        t1 = time.time()
        print(f'Build took {t1 - t0:.3f}s')
        remote.commit()

        for use_heuristic in [False, True]:
            config = GremliteConfig()
            config.use_basic_heuristic_filtering = use_heuristic
            remote = SQLiteConnection(path, config=config)
            g = traversal().with_remote(remote)

            # Make queries
            times = []
            name_sets = []
            for k in range(Q):
                last_name = random.choice(last_names)
                t0 = time.time()
                result = g.V().has('last', last_name).has('first', TextP.lt('8')).values('first').to_list()
                t1 = time.time()
                times.append(t1 - t0)
                name_sets.append(result)

            # Results:
            print()
            print(f'Used filtering heuristic: {use_heuristic}')
            print(f'Made {Q} queries.')
            print(f'Average query time: {sum(times)/len(times):.6f}s')

            verbose = False
            if verbose:
                for t, names in zip(times, name_sets):
                    print()
                    print(f'Time: {t:.6}s')
                    print(f'Number of first names: {len(names)}')


if __name__ == "__main__":
    main()
