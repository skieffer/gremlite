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
Implements a simple adapter so that the Gremlin query language can be used with Python's sqlite3
module, for a serverless, file-based graph database.
"""

import pathlib
import sqlite3
import sys
import uuid

from gremlin_python.driver.remote_connection import RemoteTraversal
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.traversal import Bytecode

from .base import GremliteConfig, ProducerInternalResultIterator
from .errors import BadDatabase
from .logging import OpenCloseLoggingConnection, PlanLoggingConnection
import gremlite.querytools as qt
from .steps import bytecode_to_producer_chain


# At least due to our use of keyword "RETURNING" (it is used in many places
# in the `querytools.py` module), we need SQLite 3.35 or later.
REQUIRED_SQLITE_MINIMUM_VERSION = (3, 35)


class InsufficientSQLiteVersionException(Exception):

    def __init__(self, detected_version, extra_phrase=None):
        self.detected_version = detected_version
        self.extra_phrase = extra_phrase or ' '

    def __str__(self):
        major, minor = REQUIRED_SQLITE_MINIMUM_VERSION
        msg = f'GremLite requires SQLite {major}.{minor}.x or later'
        msg += f', but version {self.detected_version}{self.extra_phrase}was detected.'
        return msg


def check_sqlite_version(config: GremliteConfig):
    """
    Check that we are using an acceptable version of SQLite.

    :param config: this is used for testing purposes
    :return: nothing
    :raises: InsufficientSQLiteVersionException
    """
    version_string = sqlite3.sqlite_version

    if config.phony_testing_sqlite_version_string is not None:
        version_string = config.phony_testing_sqlite_version_string

    parts = version_string.split('.')

    # If version not in three parts, we don't know how to read it:
    if len(parts) != 3:
        raise InsufficientSQLiteVersionException(
            version_string, extra_phrase=' of unrecognized format '
        )

    # I hope at least major and minor are always integers, but just in case:
    try:
        major, minor = [int(p) for p in parts[:2]]
    except ValueError:
        raise InsufficientSQLiteVersionException(
            version_string, extra_phrase=' with non-integer component(s) '
        )

    if (major, minor) < REQUIRED_SQLITE_MINIMUM_VERSION:
        raise InsufficientSQLiteVersionException(version_string)


def get_g(db_file_path,
          autocommit=None, isolation_level="DEFERRED", timeout=5.0,
          log_plans=False, session=None, config: GremliteConfig = None):
    """
    Convenience function to get a Gremlin graph traversal source for a given
    SQLite database file.

    :return: GraphTraversalSource
    """
    remote = SQLiteConnection(
        db_file_path,
        autocommit=autocommit, isolation_level=isolation_level, timeout=timeout,
        log_plans=log_plans, session=session, config=config
    )
    g = traversal().with_remote(remote)
    return g


class SQLiteConnection:
    """
    Stand-in for the `DriverRemoteConnection` class of gremlinpython.

    Whereas ordinary usage of gremlinpython to connect to an actual Gremlin server
    might look like this:

        uri = 'ws://localhost:8182/gremlin'
        remote = DriverRemoteConnection(uri)
        g = traversal().with_remote(remote)

    usage of this `SQLiteConnection` class instead looks like this:

        path = '/filesystem/path/to/my_sqlite_database_file.db'
        remote = SQLiteConnection(path)
        g = traversal().with_remote(remote)

    References
    ==========

    gremlinpython:
        https://pypi.org/project/gremlinpython/

    Gremlin:
        https://tinkerpop.apache.org/gremlin.html
        https://tinkerpop.apache.org/docs/current/reference/
        https://arxiv.org/abs/1508.03843

    SQLite:
        https://sqlite.org
        https://docs.python.org/3/library/sqlite3.html

    """

    def __init__(self, db_file_path,
                 autocommit=None, isolation_level="DEFERRED", timeout=5.0,
                 log_plans=False, check_qqc_patterns=False, log_open_close=0,
                 session=None, config: GremliteConfig = None):
        """
        :param db_file_path: string or pathlib.Path pointing to the file you want
            to use for your on-disk database with SQLite. The file can be a pre-existing one, from a
            previous use of GremLite, or else should not yet exist (for an initial use of GremLite).
            In all cases, the directory named by the path must already exist.
        :param autocommit: Set to a boolean value to control transactions as in Python 3.12 and forward.
            See https://docs.python.org/3.12/library/sqlite3.html#transaction-control
            To be clear, you can use the ``autocommit`` kwarg here even if you are using a version of
            Python prior to 3.12. This library interprets it for you.
        :param isolation_level: If not setting the ``autocommit`` kwarg, use this to control transactions
            as in Python versions prior to 3.12.
            See https://docs.python.org/3.8/library/sqlite3.html#controlling-transactions
            At time of writing, Python versions 3.12 and later also continue to support ``isolation_level``.
        :param timeout: number of seconds to wait to acquire transaction locks (default 5.0 s)
        :param log_plans: set True to record query plans via INFO-level logging
        :param check_qqc_patterns: set True to raise an exception if any unexpected "quad query constraint"
            pattern is used. This is mainly for development and diagnostics.
        :param log_open_close: set to 1 to keep track of open cursors and make their stack traces inspectible
            via gremlite.logging.print_open_cursor_traces(). Set to 2 to also log open/close events via INFO-level
            logging. This param does nothing if `log_plans` is True.
        :param session: used internally to start transactions
        :param config: optionally pass a `GremliteConfig` to make settings.
        """
        self.db_file_path = pathlib.Path(db_file_path)

        self.is_logging = log_plans is not False
        if check_qqc_patterns:
            qt.check_qqc_patterns()

        self.__session = session
        self.config = config or GremliteConfig()
        check_sqlite_version(self.config)

        connection_constructor = sqlite3.Connection
        if log_plans:
            connection_constructor = PlanLoggingConnection
        elif log_open_close > 0:
            connection_constructor = lambda database, timeout=5.0: OpenCloseLoggingConnection(
                database, timeout=timeout, log_level=log_open_close)

        self.con = connection_constructor(self.db_file_path, timeout=timeout)

        """
        The meaning of the Connection's `isolation_level` property is a bit subtle.
        To understand it, we have to first be clear that SQLite *itself* has its own
        behavior, while the Python sqlite3 *module* has *its* own behavior, and these
        are two different things.
        
        The default behavior of SQLite, independent of Python, is autocommit mode. This means
        that *if you have not explicitly said BEGIN to start a transaction* then any
        statement automatically initiates a transaction,
        *and* this transaction is automatically committed as soon as the statement is
        completed. See:
            https://sqlite.org/lang_transaction.html
            https://system.data.sqlite.org/index.html/raw/419906d4a0043afe2290c2be186556295c25c724
        
        On the contrary, the default behavior you will see if you use SQLite via Python's
        sqlite3 module is different. Like SQLite's built-in behavior, a transaction is
        automatically *begun* for any DML statement (INSERT/UPDATE/DELETE/REPLACE)
        if you have not already explicitly
        started one earlier using `BEGIN`; however, *unlike* SQLite's built-in behavior,
        this transaction is *not* automatically *committed*. So: begun yes, committed no.
        You can carry out a sequence of DML statements, and their effects will pile up in
        the transaction that Python silently started for you, but none of these effects will
        be committed until you explicitly call the `commit()` method on your `Connection`.
        See https://docs.python.org/3.8/library/sqlite3.html#controlling-transactions
        
        If you accept the default values `autocommit=None, isolation_level="DEFERRED"` of this
        `SQLiteConnection.__init__()` kwargs, you are going to get the Python sqlite3 module's
        default behavior.
        
        To recover SQLite's native autocommit behavior, you can set the `Connection` object's
        `isolation_level` attribute to `None`. Or, in Python 3.12 or later, it now has an
        `autocommit` attribute, which you can set to `True` (or you can continue to use the
        `isolation_level`, which is still there too).
        """

        if isinstance(autocommit, bool):
            vi = sys.version_info
            if vi.major >= 3 and vi.minor >= 12:
                self.con.autocommit = autocommit  # pragma: no cover
            else:
                self.con.isolation_level = None if autocommit else "DEFERRED"
        else:
            self.con.isolation_level = isolation_level

        if self.is_session_bound():
            cur = self.con.cursor()
            cur.execute('BEGIN TRANSACTION;')
            cur.close()

        if not db_already_initialized(self.con):
            initialize_db(self.con)

        self.__spawned_sessions = []

    def table_stats(self):
        return qt.get_table_stats(self.con)

    def close(self):
        if len(self.__spawned_sessions) > 0:
            for spawned_session in self.__spawned_sessions:
                spawned_session.close()
            self.__spawned_sessions.clear()

        self.con.close()

    def submit(self, bytecode: Bytecode) -> RemoteTraversal:
        """
        As in the `DriverRemoteConnection` class of gremlinpython, the `submit()`
        method is the one that is invoked through `Traversal.__next__()` when it
        calls `apply_strategies()`. This is where it is our job to take the bytecode
        representing a Gremlin traversal, and construct an iterator producing the
        results of the query.
        """
        producer = bytecode_to_producer_chain(self.config, self.con, bytecode)

        # This is purely for testing purposes:
        if self.config.traversal_returns_internal_result_iterator:
            producer = ProducerInternalResultIterator(producer)

        return RemoteTraversal(producer)

    def is_session_bound(self):
        return self.__session is not None

    def create_session(self):
        """
        Supports the `gremlin_python.process.graph_traversal.Transaction.begin()` method.
        """
        if self.is_session_bound():
            raise Exception(
                'Connection is already bound to a session - child sessions are not allowed')  # pragma: no cover
        con = SQLiteConnection(self.db_file_path, log_plans=self.is_logging, session=uuid.uuid4())
        self.__spawned_sessions.append(con)
        return con

    def remove_session(self, session_based_connection):
        session_based_connection.close()
        self.__spawned_sessions.remove(session_based_connection)

    def commit(self):
        self.con.commit()

    def rollback(self):
        self.con.rollback()


# These are the indexes that we use, on the quads table:
quad_index_names = """
ops
sp
pg
g
""".split()


def initialize_db(con: sqlite3.Connection):
    """
    Create the tables and indexes that we need.

    Data Model
    ==========

    We use a variant of the "RDF quads" design for recording a property graph in a 4-column table in a relational
    database, as described in

        Harth, Andreas, and Stefan Decker. "Optimized index structures for
        querying RDF from the web." In Third Latin American Web Congress
        (LA-WEB'2005), pp. 10-pp. IEEE, 2005.

        https://ieeexplore.ieee.org/abstract/document/1592360

    See also:

        https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview-data-model.html

    Indexes
    -------

    We use a different set of indexes than those suggested by either of the two above references, namely,
    we use the four indexes: ops, sp, pg, g.

    To begin with, while those sources consider each column's constraints to be binary (on or off), for us they are
    ternary: exact (equal to a particular value or in a finite set of values), sign (pos. or neg.), or none. This is
    important because in order for SQLite to make use of an index, the columns with exact constraints have to come
    before those with sign constraints. As stated on this page: https://www.sqlite.org/optoverview.html,
    "The initial columns of the index must be used with the = or IN or IS operators. The right-most column that is
    used can employ inequalities."

    We describe a query to the quads table by a pattern like "[po|s]", which means that we are putting exact
    constraints on the p and o columns, a sign constraint on the s column, and no constraint on the g column.

    At the time of developing our first release of this project, we performed a survey of all the quad constraint
    patterns that were actually used in all of the Gremlin steps we were supporting at that time. (Our unit tests
    record them in a log, and we have code to survey the log.) This resulted in the list of patterns recorded in
    the `expected_quad_constraint_patterns` in the `querytools.py` module.

                                                [|g]        (g)
                                                [s|p]       (sp)
                                                [s|]        (sp)
                                                [sp|]       (sp)
                                                [spo|]      (ops)
                                                [p|g]       (pg)
                                                [p|]        (pg)
                                                [po|s]      (ops)
                                                [po|]       (ops)
                                                [pg|]       (pg)
                                                [o|p]       (ops)
                                                [g|]        (g)

                            The expected query patterns, and the indexes that cover them.

    After studying this list we chose a set of four indexes that cover all these patterns. (For an index to cover
    a pattern [L|R] it has to be possible to choose a permutation of L, and a permutation of R, and concatenate these,
    and thereby get some prefix of the index description.) Note that, among our expected patterns, each of the four
    column letters s, p, o, g occurs alone on the left-hand side of at least one pattern, which means that we need
    *at least* four indexes: one to support each column as the sole exact constraint. Therefore it seems plausible
    that our chosen list of four indexes might be optimal (though I can't say for sure).

    Our unit tests contain checks (a) that no unexpected quad constraint patterns are used, and (b) that all
    expected ones are used. If check (a) should fail as we move forward and support more of the Gremlin
    language, it may mean that our new code should be reworked to stay within the set of expected patterns, or
    it may mean that we need a new index. But the latter should be avoided if possible, since indexes impose both space
    costs (file size) and time costs when writing to the database. If check (b) should fail, it would mean that we
    had already reworked things somehow, and should reevaluate whether we still have the best set of indexes.

    IDs, Valence, and Property Value Encodings
    ------------------------------------------

    ID numbers are assigned to three types of "graph elements:" vertices, edges, and vertex properties.
    (Edge properties are not treated the same way. Everything we do is modeled on the behavior of TinkerGraph 3.7.2.)
    These three types of graph elements share a single ID space of positive integers. Thus, no two of these will
    ever have the same ID. ID numbers are never reused after objects are deleted from the graph.

    ID numbers are also assigned to strings, which encompass vertex and edge labels, property names, and string or
    float property values. These too are auto-incrementing positive integers, but they start at 3 (in order to
    make room for encoding property values of other types, including boolean and null).

    When IDs are recorded in the table of quads, they are sometimes negated, as indicated below, in order to resolve
    any possible ambiguities. We refer to the sign of an ID when recorded in the quads table as its "valence."
    To be clear, all IDs themselves are positive integers; they are only sometimes *recorded* with negative sign.

        For the S column ("subject"):

            edges: < 0
            vertices: > 0

        For the P column ("predicate"):

            edge labels: < 0
            "has label": = 0
            property names: > 0

        For the O column ("object"):

            vertex labels: > 0

            vertices: > 0

            integer: negative (add 2^62 to recover actual value)
            boolean: 0 or 1 for False or True resp.
            null: 2
            strings & floats: >= 3. When greater than 2^62, interpreted as float.

        For the G column (edge ID or vertex property ID):

            edges: < 0
            vertex properties: > 0

    Integer property values must lie in the range [-2^62, 2^62 - 1]. They are shifted by -2^62
    before being recorded (thus, are always negative in the table).

    The "O" column is the only one in which different types share certain integer ranges, but
    the value in the "P" column always disambiguates it:

        * O can only be "vertices" if P < 0 ("edge labels")
        * O can only be "vertex labels" if P = 0 ("has label")
        * O can only be "int/bool/null/string/float" if P > 0 ("property names")

    Note that column "O" never holds edge labels because those are recorded differently;
    namely, they go in the "P" column in quads where the "G" column holds the (negated) edge ID.

    Quad Types
    ----------

    There are only four different types of quads in the quads table, which correspond to
    the four different types of things that there are in a property graph: vertices, edges,
    vertex properties, and edge properties.

    From the encoding described above you can figure out for yourself what the four quad types
    look like, but we summarize them here for convenience.

    Vertex:

        (+vertex ID, 0, +vertex label ID, 0)

    Edge:

        (+source vertex ID, -edge label ID, +target vertex ID, -edge ID)

    Vertex Property:

        (+vertex ID, +property name ID, encoded property value, +vertex property ID)

    Edge Property:

        (-edge ID, +property name ID, encoded property value, 0)

    It follows that each quad type can be recognized immediately by a single column condition:

        Vertex: p = 0
        Edge:   p < 0 or g < 0 (either condition on its own is sufficient)
        Prop:   p > 0 (this identifies a property of any kind, vertex or edge)
        V Prop: g > 0
        E Prop: s < 0

    """
    cur = con.cursor()

    cur.execute("CREATE TABLE quads (s INT, p INT, o INT, g INT)")

    for name in quad_index_names:
        cur.execute(f"CREATE INDEX {name} ON quads ({', '.join(list(name))})")

    cur.execute("CREATE TABLE strings (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT UNIQUE)")
    # Note: There is no need to create an index to look up string IDs by string values,
    # since SQLite will just use its own internal index `sqlite_autoindex_strings_1` for that anyway,
    # which it forms in order to enforce the UNIQUE constraint.
    # See https://stackoverflow.com/a/35626972

    # Ensure that string IDs start at 3:
    cur.execute("INSERT INTO strings (v) VALUES ('foo')")
    cur.execute("INSERT INTO strings (v) VALUES ('bar')")
    cur.execute("DELETE FROM strings WHERE id = 1")
    cur.execute("DELETE FROM strings WHERE id = 2")
    # In fact 3 will always be the ID of the empty string. We want ID 3 to actually be
    # present, so that we can make a positive check in the `db_already_initialized()` function.
    cur.execute("INSERT INTO strings (v) VALUES ('')")

    # For graph elements (vertices, edges, and vertex properties) there is nothing to record except their ID.
    # So this table just serves to record the set of existing IDs, as well as to produce new IDs.
    # We use an AUTOINCREMENT id column instead of built-in rowid, to ensure
    # that ids are never reused. See https://stackoverflow.com/a/9342301
    cur.execute("CREATE TABLE graphelts (id INTEGER PRIMARY KEY AUTOINCREMENT)")

    cur.close()
    con.commit()


expected_table_names = {'quads', 'strings', 'graphelts'}

# We include a check that sqlite's 'sqlite_autoindex_strings_1' is formed, because we are relying on that.
expected_index_names = set(quad_index_names).union({'sqlite_autoindex_strings_1'})


def db_already_initialized(con: sqlite3.Connection):
    """
    Check if a database has been initialized, with exactly the tables
    and indexes that we expect.

    :param con: connection
    :return: boolean, True if it looks like the database has already been initialized,
        False if it looks brand new.
    :raise: BadDatabase if there are one or more tables or indexes, but they are not
        as we expect. This should only happen if you connected to some database
        used for some other purpose than ours.
    """
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        actual_table_names = {r[0] for r in cur.fetchall()}

        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        actual_index_names = {r[0] for r in cur.fetchall()}

        # Among the actual tables and indices, there will be special ones like 'sqlite_sequence'.
        # We don't want to worry about those, so we just check that all expected ones are present.
        if expected_table_names.issubset(actual_table_names) and expected_index_names.issubset(actual_index_names):

            # Check that string IDs start at 3
            R = cur.execute("SELECT rowid FROM strings WHERE rowid <= 3").fetchall()
            if len(R) != 1 or R[0][0] != 3:
                raise BadDatabase

            return True
        elif len(actual_table_names) == 0 and len(actual_index_names) == 0:
            return False
        else:
            raise BadDatabase
    finally:
        cur.close()
