"""
Test the logging of open/close cursor events.
"""

import logging
import pathlib
import sqlite3
import tempfile

from gremlin_python.process.anonymous_traversal import traversal

from gremlite import SQLiteConnection, GremliteConfig
from gremlite.logging import print_open_cursor_traces
from tests.test_steps import make_random_graph


def main(log_level):
    """
    The test we perform here is modeled after our existing unit test called
    `test_locked_database()`, in the `tests.test_steps` module.

    Here we have it as a standalone script, so that we can easily experiment
    with the different log levels.

    :param log_level: will be passed to `log_open_close` in the connections we
        form, i.e. will govern the level of cursor open/close logging that we get.
    """
    logging.basicConfig(filename='cursor_logging.log', level=logging.INFO)
    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'

        conf = GremliteConfig()
        conf.read_all_at_once = True

        # Form two connections to a single database:
        remote1 = SQLiteConnection(path, log_open_close=log_level, config=conf)
        g1 = traversal().with_remote(remote1)

        # Give this one a fast timeout so we don't have to wait:
        remote2 = SQLiteConnection(path, timeout=0.05, log_open_close=log_level, config=conf)
        g2 = traversal().with_remote(remote2)

        def start_reading(remote):
            c = remote.con.cursor()
            c.execute('SELECT * FROM quads')
            return c

        Nv, Ne = 50, 70

        # Build using conn 1:
        make_random_graph(g1, Nv, Ne)
        remote1.commit()

        # Conn 2 sees the built graph:
        assert g2.V().count().next() == Nv

        # Conn 1 begins reading:
        cur = start_reading(remote1)

        # Conn 2 cannot drop the graph.
        # Note that we *can* carry out the drop traversal; the error does not actually
        # occur until we try to *commit* on the second connection.
        g2.V().drop().iterate()
        try:
            remote2.commit()
        except sqlite3.OperationalError:
            print_open_cursor_traces()

        # Once conn 1 closes its reading cursor, then conn 2 can write.
        cur.close()
        remote2.commit()
        # Conn 1 sees that the graph has been dropped:
        assert g1.V().count().next() == 0


if __name__ == "__main__":
    log_level = 1
    main(log_level)
