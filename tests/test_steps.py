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

import pathlib
import random
import sqlite3
import tempfile

import pytest

from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import Cardinality, Direction, Order, P, T, TextP
import gremlin_python.structure.graph as graph

from tests.cull_log import cull_log
from gremlite.connection import (
    SQLiteConnection, db_already_initialized, expected_index_names,
    GremliteConfig, get_g, quad_index_names, InsufficientSQLiteVersionException,
)
from gremlite.errors import (
    UnsupportedUsage, BadStepArgs, BadStepCombination,
    BadDatabase, UnexpectedQuadConstraintPattern,
)
import gremlite.querytools as qt
from gremlite.results import Result


TABLE_NAMES = ['quads', 'strings', 'graphelts']


def get_basic_setup(dirname, filename='foo.db',
                    log_plans=True, check_qqc_patterns=True,
                    air_routes=True, timeout=5.0,
                    config=None):
    """
    Start a new graph database, build the mini air routes graph in it, and
    return both g and remote.
    """
    path = pathlib.Path(dirname) / filename
    remote = SQLiteConnection(path,
                              log_plans=log_plans, check_qqc_patterns=check_qqc_patterns,
                              timeout=timeout, config=config)
    g = traversal().with_remote(remote)
    if air_routes:
        make_mini_air_routes_graph(g)
        remote.commit()
    return g, remote


def get_TinkerGraph_g():
    """
    Sometimes in testing it is useful to be able to check behavior against
    our reference implementation, TinkerGraph. For that, you can start up
    a server with

        $ docker run --rm -p 8182:8182 tinkerpop/gremlin-server:3.7.2

    and then this function will give you a `g` connected to that server.

    :return: GraphTraversalSource g
    """
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    uri = 'ws://localhost:8182/gremlin'
    remote = DriverRemoteConnection(uri)
    g = traversal().with_remote(remote)
    return g


def dump_table(remote: SQLiteConnection, table_name: str):
    cur = remote.con.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    print(f'TABLE: {table_name}')
    print('-' * 40)
    rows = cur.fetchall()
    cur.close()
    print(rows)
    for row in rows:
        print(row)
    return rows


def dump_all_tables(remote: SQLiteConnection):
    for table_name in TABLE_NAMES:
        print()
        dump_table(remote, table_name)


def paths_to_lists(list_of_paths):
    return [path.objects for path in list_of_paths]


# The mini air routes graph looks like this:
#
#         ,------------
#        v             \
#     AUS ---> DFW <--> LAX
#      \      ^  \      /
#       v    /    v    v
#        ATL ----> JFK
#
# See https://kelvinlawrence.net/book/Gremlin-Graph-Guide.html#testgraph
def make_mini_air_routes_graph(g):
    (g.add_v("airport").property("code", "AUS").as_("aus").
     add_v("airport").property("code", "DFW").as_("dfw").
     add_v("airport").property("code", "LAX").as_("lax").
     add_v("airport").property("code", "JFK").as_("jfk").
     add_v("airport").property("code", "ATL").as_("atl").
     add_e("route").from_("aus").to("dfw").
     add_e("route").from_("aus").to("atl").
     add_e("route").from_("atl").to("dfw").
     add_e("route").from_("atl").to("jfk").
     add_e("route").from_("dfw").to("jfk").
     add_e("route").from_("dfw").to("lax").
     add_e("route").from_("lax").to("jfk").
     add_e("route").from_("lax").to("aus").
     add_e("route").from_("lax").to("dfw").iterate())


def make_random_graph(g, Nv=100, Ne=200):
    """
    Build a random graph.

    :param g: traversal source
    :param Nv: number of vertices
    :param Ne: number of edges
    :return: nothing
    """
    vertex_ids = []
    for i in range(Nv):
        vid = g.add_v('vType').id_().next()
        vertex_ids.append(vid)

    for i in range(Ne):
        vid1, vid2 = random.choices(vertex_ids, k=2)
        g.V(vid2).as_('v2').V(vid1).add_e('eType').to('v2').iterate()


def test_db_already_initialized():
    """
    Check that we can reconnect to a db file that has already been initialized.
    """
    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'

        remote = SQLiteConnection(path)
        remote.close()

        remote2 = SQLiteConnection(path)
        assert db_already_initialized(remote2.con)
        remote2.close()


def test_make_mini_air_routes_graph():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        #dump_all_tables(remote)

        expected_quads = [(1, 0, 4, 0), (1, 5, 6, 2), (3, 0, 4, 0), (3, 5, 7, 4),
                          (5, 0, 4, 0), (5, 5, 8, 6), (7, 0, 4, 0), (7, 5, 9, 8),
                          (9, 0, 4, 0), (9, 5, 10, 10),
                          (1, -11, 3, -11), (1, -11, 9, -12), (9, -11, 3, -13), (9, -11, 7, -14),
                          (3, -11, 7, -15), (3, -11, 5, -16),
                          (5, -11, 7, -17), (5, -11, 1, -18), (5, -11, 3, -19)]

        expected_strings = [(3, ''), (4, 'airport'), (5, 'code'), (6, 'AUS'), (7, 'DFW'),
                            (8, 'LAX'), (9, 'JFK'), (10, 'ATL'), (11, 'route')]

        expected_graphelts = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,),
                              (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,)]

        expected = [
            expected_quads, expected_strings, expected_graphelts
        ]

        for name, exp in zip(TABLE_NAMES, expected):
            print()
            actual_rows = dump_table(remote, name)
            assert actual_rows == exp

        rc = remote.table_stats()
        print('\n', rc)
        assert str(rc) == '(qe: 9, qvl: 5, qvp: 5, qep: 0, gv: 0, ge: 0, gvp: 0, s: 9)'


@pytest.mark.parametrize(['phony_version', 'expected_error'], [
    # This first case checks the actual SQLite version on the machine where the unit
    # tests are running. If they are to pass as a whole, this has to be a sufficient version!
    [None, None],
    # For the other cases, we try a series of phony versions, to cover the various possibilities.
    ['4.0.0', None],
    ['3.36.10-alpha', None],
    ['3.35.0', None],
    ['3.34.17', InsufficientSQLiteVersionException('3.34.17')],
    ['2.3.5', InsufficientSQLiteVersionException('2.3.5')],
    ['1.2', InsufficientSQLiteVersionException('1.2', extra_phrase=' of unrecognized format ')],
    ['3.foo.7', InsufficientSQLiteVersionException('3.foo.7', extra_phrase=' with non-integer component(s) ')]
])
def test_check_sqlite_version(phony_version, expected_error):
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        if phony_version is not None:
            config.phony_testing_sqlite_version_string = phony_version

        if expected_error is None:
            get_basic_setup(dirname, config=config)
        else:
            with pytest.raises(InsufficientSQLiteVersionException) as e:
                get_basic_setup(dirname, config=config)
            assert str(e.value) == str(expected_error)


def test_get_g():
    with tempfile.TemporaryDirectory() as dirname:
        filename = 'foo.db'
        get_basic_setup(dirname, filename=filename)
        g1 = get_g(pathlib.Path(dirname) / filename)
        result = g1.V().count().next()
        assert result == 5


def test_bad_args():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        with pytest.raises(BadStepArgs):
            # key can't be float
            g.V().has(3.14, 'foo').iterate()

        with pytest.raises(BadStepArgs):
            # value must be acceptable property value type
            g.V().has('foo', {}).iterate()

        with pytest.raises(BadStepArgs):
            # `as()` step needs at least one arg
            g.V(1).as_().iterate()

        with pytest.raises(BadStepArgs):
            # `add_v()` step needs either 0 or 1 args
            g.add_v('foo', 'bar').iterate()

        with pytest.raises(BadStepArgs):
            # `store()` needs an arg
            g.V(1).id_().store().iterate()

        with pytest.raises(BadStepArgs):
            # `count()` needs no args
            g.V().count('foo').next()

        with pytest.raises(BadStepArgs):
            # `to()` needs a string arg
            g.V(1).add_e('foo').to(3.14).iterate()

        with pytest.raises(BadStepCombination):
            # `add_e()` needs an incoming vertex
            g.E(11).add_e('foo').iterate()

        with pytest.raises(BadStepArgs):
            # First of three args to `property()` must be cardinality
            g.V(1).property('foo', 'bar', 'spam').iterate()

        with pytest.raises(BadStepArgs):
            # `property()` can't accept more than 3 args
            g.V(1).property('foo', 'bar', 'spam', 'baz').iterate()

        with pytest.raises(BadStepArgs):
            # `property()` can't accept less than 2 args
            g.V(1).property('foo').iterate()

        with pytest.raises(BadStepArgs):
            # `property()` name must be string
            g.V(1).property(3.14, 'bar').iterate()

        with pytest.raises(BadStepArgs):
            # `property()` value must be acceptable type
            g.V(1).property('foo', {}).iterate()

        with pytest.raises(BadStepArgs):
            # `constant()` needs exactly 1 arg
            g.V().constant(1, 2).iterate()

        with pytest.raises(BadStepArgs):
            # `constant()` needs exactly 1 arg
            g.V().constant().iterate()

        with pytest.raises(NotImplementedError):
            # `constant()` does not accept bytecode
            g.V().constant(__.V()).iterate()


@pytest.mark.parametrize(['use_tx', 'ac', 'iso_lev', 'nv2'], [
    # ------------------
    # No transactions.
    # When using Python sqlite3's transaction management, a second connection will
    # see nothing until we call commit() on the first connection.
    [False, None, "DEFERRED", 0],
    [False, False, "DEFERRED", 0],
    # When using SQLite's autocommit mode, a second connection *will* see everything,
    # even before doing a (now superfluous) commit on the first.
    [False, None, None, 5],
    [False, True, None, 5],
    # ------------------
    # With transactions.
    # Now it doesn't matter whether we're in SQLite's autocommit mode, or Python sqlite3's
    # transaction management mode. Either way, by explicitly starting our own transaction tx,
    # we have made it so that nothing will be committed until we call commit on tx.
    [True, None, "DEFERRED", 0],
    [True, False, "DEFERRED", 0],
    [True, None, None, 0],
    [True, True, None, 0],
])
def test_transactions(use_tx, ac, iso_lev, nv2):
    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'

        remote1 = SQLiteConnection(path, autocommit=ac, isolation_level=iso_lev, log_plans=True)
        g1 = traversal().with_remote(remote1)

        if use_tx:
            tx = g1.tx()
            g1 = tx.begin()
            committer = tx
        else:
            committer = remote1

        # First connection builds the graph but does not commit.
        make_mini_air_routes_graph(g1)
        # All vertices are visible to this connection:
        result1 = g1.V().count().next()
        assert result1 == 5

        # What does a second connection see?
        remote2 = SQLiteConnection(path, log_plans=True)
        g2 = traversal().with_remote(remote2)
        result2 = g2.V().count().next()
        assert result2 == nv2

        # *Now* the first connection commits, and the second connection looks again.
        # In all cases, it now sees the vertices.
        committer.commit()
        result2 = g2.V().count().next()
        assert result2 == 5


def test_transaction_close_and_rollback():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # We can rollback.
        tx = g.tx()
        tx.begin()
        remote.rollback()

        # Can close connection while a transaction is open.
        g.tx()
        remote.close()


def test_bad_db():
    with tempfile.TemporaryDirectory() as dirname:
        dirpath = pathlib.Path(dirname)

        # DB does not have the expected tables.
        db_path = dirpath / 'foo.db'

        con = sqlite3.Connection(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE foo (bar INT)")
        cur.close()
        con.commit()

        with pytest.raises(BadDatabase):
            get_g(db_path)

        # DB has the expected tables but strings table doesn't start at ID 3
        db_path = dirpath / 'bar.db'

        con = sqlite3.Connection(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE quads (s INT, p INT, o INT, g INT)")
        for name in quad_index_names:
            cur.execute(f"CREATE INDEX {name} ON quads ({', '.join(list(name))})")
        cur.execute("CREATE TABLE strings (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT UNIQUE)")
        cur.execute("INSERT INTO strings (v) VALUES ('foo')")
        cur.execute("CREATE TABLE graphelts (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        cur.close()
        con.commit()

        with pytest.raises(BadDatabase):
            get_g(db_path)


def test_removal_methods_directly():
    """
    Here we try using the graph element removal methods directly.
    See also `test_drop()`.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        con = remote.con

        # Give the edges some properties, so we can check their removal:
        g.E().property('time', 1).iterate()

        #####################################################################
        # Remove a vertex
        dfw_id = g.V().has('code', 'DFW').id_().next()

        dump_table(remote, 'quads')

        rc = qt.completely_remove_vertex(con, dfw_id)
        print('\n', rc)

        # Got expected numbers of deleted rows:
        assert rc.matches((5, 1, 1, 5, 1, 5, 1, 0))

        dump_table(remote, 'quads')

        cur = con.cursor()
        all_quads = cur.execute("SELECT * FROM quads").fetchall()
        cur.close()

        # DFW is no longer a subject at all, hence, has no:
        #  - outgoing edges
        #  - label
        #  - properties
        assert all(q[0] != dfw_id for q in all_quads)

        # DFW is no longer object when p < 0, hence, has no:
        #   - incoming edges
        assert all(q[1] >= 0 or q[2] != dfw_id for q in all_quads)

        #####################################################################
        # Remove one of the remaining edges

        e_id = g.E().and_(__.out_v().has('code', 'LAX'), __.in_v().has('code', 'AUS')).id_().next()
        neg_e_id = -e_id
        rc = qt.completely_remove_edges(con, neg_e_id)
        print('\n', rc)
        assert rc.matches((1, 0, 0, 1, 0, 1, 0, 0))

        #####################################################################
        # Remove one of the remaining vertex properties

        lax_code_prop_id = g.V().has('code', 'LAX').properties('code').id_().next()
        rc = qt.completely_remove_vertex_property(con, lax_code_prop_id)
        print('\n', rc)
        assert rc.matches((0, 0, 1, 0, 0, 0, 1, 0))

        #####################################################################
        # Remove one of the remaining edge properties

        e_id = g.E().and_(__.out_v().has('code', 'ATL'), __.in_v().has('code', 'JFK')).id_().next()
        rc = qt.completely_remove_edge_property(con, e_id, 'time')
        print('\n', rc)
        assert rc.matches((0, 0, 0, 1, 0, 0, 0, 0))

        dump_table(remote, 'quads')
        dump_table(remote, 'graphelts')


def test_drop():
    """
    Here we test removal of graph elements using the `drop()` step.
    See also `test_removal_methods_directly()`.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Give the edges some properties, so we can check their removal:
        g.E().property('time', 1).iterate()

        # Drop a vertex:
        g.V().has('code', 'DFW').drop().iterate()
        # Drop an edge:
        g.E().and_(__.out_v().has('code', 'LAX'), __.in_v().has('code', 'AUS')).drop().iterate()
        # Drop a vertex property:
        g.V().has('code', 'LAX').properties('code').drop().iterate()
        # Drop an edge property:
        g.E().and_(__.out_v().has('code', 'ATL'), __.in_v().has('code', 'JFK')).properties('time').drop().iterate()

        expected_quads = [(1, 0, 4, 0), (1, 5, 6, 2), (5, 0, 4, 0), (7, 0, 4, 0), (7, 5, 9, 8),
                          (9, 0, 4, 0), (9, 5, 10, 10),
                          (1, -11, 9, -12), (9, -11, 7, -14), (5, -11, 7, -17),
                          (-17, 12, -4611686018427387903, 0), (-12, 12, -4611686018427387903, 0)]

        expected_graphelts = [(1,), (2,), (5,), (7,), (8,), (9,), (10,), (12,), (14,), (17,)]

        expected = [expected_quads, expected_graphelts]

        for name, exp in zip(['quads', 'graphelts'], expected):
            print()
            actual_rows = dump_table(remote, name)
            assert actual_rows == exp

        # Delete the whole graph
        g.V().drop().iterate()
        assert g.V().count().next() == 0
        assert g.E().count().next() == 0


def test_property_filter_obj_type():
    """
    Because properties are encoded in the quads table in the form,

        (signed_subject_id, prop_name_id, prop_value_id, 0 or vertex property ID)

    the sign of the subject id becomes critical in cases where both a vertex and an edge
    may have the same unsigned id and same property value. This test confirms that in such
    a case, we can properly locate the vertex or the edge, as desired, using a property
    filter alone (i.e. no label filter, which works differently).
    """
    with tempfile.TemporaryDirectory() as dirname:
        path = pathlib.Path(dirname) / 'foo.db'
        remote = SQLiteConnection(path)
        g = traversal().with_remote(remote)

        # Make a graph having a vertex of id 1 and property foo=bar,
        # and also an edge having id 1 and property foo=bar.
        (g.add_v('vertex').property('foo', 'bar').as_('s')
         .add_v('vertex').property('foo', 'cat').as_('t')
         .add_e('edge').from_('s').to('t').property('foo', 'bar')
         .iterate())

        remote.commit()

        # To make a proper test, it's important that here we
        # do NOT first filter on label. Because of the way labels are encoded in
        # the quads table, a label filter already ensures that we are constraining
        # the s column to have the correct sign. What we want to test here is that
        # the property filter alone is enough to achieve the right sign.

        result = g.V().has('foo', 'bar').next()
        assert isinstance(result, graph.Vertex)

        result = g.E().has('foo', 'bar').next()
        assert isinstance(result, graph.Edge)


def test_qt_low_level():
    """
    Test some low-level querytools functions that happen not to be reachable at the moment,
    due to the way certain things work.

    For example, we can't currently reach the `get_edge_label()` function due to the fact that
    `Result.form_edge()` always determines the label as soon as an edge is added to the path.
    This is because we need to determine the edge's endpoints, and while we're doing that it
    makes sense to record the label, since it is returned in the same quad.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)
        con = remote.con

        assert qt.Sign.valence(qt.SubjectTypes.VERTICES) == qt.Sign.POS
        assert qt.Sign.valence(qt.SubjectTypes.EDGES) == qt.Sign.NEG

        result = qt.get_label(con, 11, qt.SubjectTypes.EDGES)
        assert result == 'route'

        assert qt.encode_property_value(con, {}) is None

        result = qt.get_predicate_column_constraint(con, P.within([11, 12, 13]), qt.ColumnRole.EDGE)
        print('\n', result)
        assert result == [-11, -12, -13]

        result = qt.get_predicate_column_constraint(con, P.within([-11, -12, -13]), qt.ColumnRole.EDGE)
        print('\n', result)
        assert result == [-11, -12, -13]

        result = qt.get_predicate_column_constraint(con, P.within([1, 3, 5]), qt.ColumnRole.VERTEX)
        print('\n', result)
        assert result == [1, 3, 5]

        result = qt.get_predicate_column_constraint(
            con, P.within(['code', 'foo', 'airport']), qt.ColumnRole.PROPERTY_NAME)
        print('\n', result)
        assert result == [5, 4]

        assert qt.decode_vertex_label(con, 4) == 'airport'

        assert qt.encode_label(con, qt.SubjectTypes.VERTICES, 3.14) is None
        assert qt.encode_label(con, qt.SubjectTypes.EDGES, 3.14) is None


def test_reuse_temp_object_label():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        g.V().has('code', 'AUS').as_('foo').V().has('code', 'DFW').as_('foo').iterate()


def test_low_level_result_obj_methods():
    """
    Test functionality of the `Result` class that is not currently covered by
    processing traversals.
    """
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        # This makes it so that traversals will produce `Result` instances:
        config.traversal_returns_internal_result_iterator = True
        g, remote = get_basic_setup(dirname, config=config)

        # cover `last_subject_type` property method, and EDGES half of
        # `get_subject_type()` method.
        r = g.E(11).next()
        assert isinstance(r, Result)
        assert r.last_subject_type is qt.SubjectTypes.EDGES

        # Make `get_subject_type()` return None
        r = g.E().count().next()
        assert isinstance(r, Result)
        assert r.last_subject_type is None

        # Cover case in copy where we don't find the sought label.
        r = g.V().has('code', 'AUS').as_('foo').next()
        assert isinstance(r, Result)
        r1 = r.copy(stop_at_latest='bar')
        assert len(r1) == 0

        # Use `has_labeled_edge()`
        r = g.V().has('code', 'AUS').out_e().as_('foo').next()
        assert isinstance(r, Result)
        assert r.has_labeled_edge('foo') is True

        # Show that an object label can't override a storage set name.
        r = g.V().id_().store('foo').V().has('code', 'AUS').as_('foo').next()
        assert isinstance(r, Result)
        obj = r.get_labeled_object('foo')
        assert obj == [1]

        # Add a vertex to the path with new label and/or properties info
        r = g.V(1).next()
        assert isinstance(r, Result)
        r.add_vertex_to_path(1, label='foo', properties={})

        # Add an edge to the path with new label and/or properties info
        r = g.E(11).next()
        assert isinstance(r, Result)
        r.add_edge_to_path(11, label='foo', properties={})


def test_odd_cases():
    """
    Test various odd cases.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Retrieve a single property value when there are multiple.
        g.V().has('code', 'JFK').property(Cardinality.set_, 'code', 'NYC').iterate()
        g.V().has('code', 'JFK').element_map().next()

        # `select()` returns nothing since given label is not present
        result = g.V().has('code', 'AUS').as_('foo').select('bar').has_next()
        assert result is False

        # `select()` returns nothing since modulator returns no results
        result = g.V().has('code', 'AUS').as_('foo').select('foo').by('bar').has_next()
        assert result is False

        # Try to use an `emit()` without a `repeat()
        with pytest.raises(BadStepCombination):
            g.V().emit().next()

        # Try to make an edge connecting to a missing vertex
        with pytest.raises(BadStepCombination):
            g.V(1).add_e('foo').to('bar').iterate()

        # Store an object by a modulator that returns no results
        result = g.V().store('foo').by('bar').cap('foo').next()
        assert result == []

        # Modulate a value_map() by a continuation that produces no results
        result = g.V(1).value_map().by(__.select('foo')).next()
        assert result == {}

        # Try to use `other_v()` when previous object is not one of the edge's endpoints.
        with pytest.raises(BadStepCombination):
            g.V(9).E(11).other_v().next()


def test_element_map():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Accept all properties
        result = g.V().has('airport', 'code', 'AUS').element_map().next()
        print()
        print(result)
        expected = {T.id: 1, T.label: 'airport', 'code': 'AUS'}
        assert result == expected

        # Limit properties
        result = g.V().has('airport', 'code', 'AUS').element_map('foo').next()
        print()
        print(result)
        expected = {T.id: 1, T.label: 'airport'}
        assert result == expected

        # Try an edge
        g.E(11).property('time', 1).iterate()
        result = g.E(11).element_map().next()
        print()
        print(result)
        expected = {
            T.id: 11, T.label: 'route',
            Direction.OUT: {T.id: 1, T.label: 'airport'},
            Direction.IN: {T.id: 3, T.label: 'airport'},
            'time': 1
        }
        assert result == expected


def test_value_map():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Let's have a couple of different property names, and a property with
        # multiple values.
        g.V().has('code', 'JFK').property('tz', 'Eastern').iterate()
        g.V().has('code', 'JFK').property(Cardinality.set_, 'code', 'NYC').iterate()

        # Basic
        result = g.V().has('code', 'JFK').value_map().next()
        assert result == {'code': ['JFK', 'NYC'], 'tz': ['Eastern']}

        # Incude tokens
        result = g.V().has('code', 'JFK').value_map(True).next()
        assert result == {T.id: 7, T.label: 'airport', 'code': ['JFK', 'NYC'], 'tz': ['Eastern']}

        # Limit to selected property names
        result = g.V().has('code', 'JFK').value_map('tz', 'foobar').next()
        assert result == {'tz': ['Eastern']}

        # Use both token and name args
        result = g.V().has('code', 'JFK').value_map(False, 'tz', 'foobar').next()
        assert result == {'tz': ['Eastern']}

        # Modulate the property lists
        result = g.V().has('code', 'JFK').value_map().by(__.unfold()).next()
        assert result == {'code': 'JFK', 'tz': 'Eastern'}


def test_simple_evaluators():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # constant()
        result = g.V().has('airport', 'code', 'AUS').constant(3.14).next()
        print()
        print(result)
        assert result == 3.14

        # id_()
        result = g.V().has('airport', 'code', 'AUS').id_().next()
        print()
        print(result)
        assert result == 1

        # label()
        result = g.V().has('airport', 'code', 'AUS').label().next()
        print()
        print(result)
        assert result == 'airport'


def test_identity():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result1 = g.V().has('airport', 'code', 'AUS').next()
        result2 = g.V().has('airport', 'code', 'AUS').identity().next()
        assert result2 == result1

        result1 = g.E(11).next()
        result2 = g.E(11).identity().next()
        assert result2 == result1


def test_count():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().count().next()
        print()
        print(result)
        assert result == 5

        result = g.E().count().next()
        print()
        print(result)
        assert result == 9

        result = g.V().has_label('foo').count().next()
        print()
        print(result)
        assert result == 0

        result = g.V().has('code', 'SEA').count().next()
        print()
        print(result)
        assert result == 0

        result = g.V().has('airport', 'code', 'AUS').out_e('route').count().next()
        print()
        print(result)
        assert result == 2


def test_coalesce():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # The first continuation is empty. We get a result from the second continuation.
        result = g.V().has('airport', 'code', 'AUS').coalesce(
            __.out().out().has('code', 'ATL').constant(1),
            __.out().has('code', 'ATL').constant(2)
        ).next()
        assert result == 2

        # Both continuations are empty. We get no results.
        result = g.V().has('airport', 'code', 'AUS').coalesce(
            __.out().out().has('code', 'ATL').constant(1),
            __.out().out().has('code', 'ATL').constant(2)
        ).has_next()
        assert result is False

        # We read all the values from the second continuation.
        # There are four results, of which three are distinct.
        result = g.V().has('airport', 'code', 'AUS').coalesce(
            __.out().out().has('code', 'ATL'),
            __.out().out().values('code')
        ).to_list()
        assert len(result) == 4
        assert set(result) == {'DFW', 'JFK', 'LAX'}

        # Even when more than one traversal produces results, we stop after
        # exhausting the first productive one. That's why we get 4 results
        # here, not 8.
        result = g.V().has('airport', 'code', 'AUS').coalesce(
            __.out().out().has('code', 'ATL'),
            __.out().out().values('code'),
            __.out().out().values('code')
        ).to_list()
        assert len(result) == 4

        # To confirm, we replace `coalesce` with `union` in the above test, and
        # see that then not just 4, but 8 results are produced.
        result = g.V().has('airport', 'code', 'AUS').union(
            __.out().out().has('code', 'ATL'),
            __.out().out().values('code'),
            __.out().out().values('code')
        ).to_list()
        assert len(result) == 8


def test_filtering_strategies():
    """
    The purpose of this test is to examine the different ways of handling `has()` steps,
    as controlled by the config var `use_basic_heuristic_filtering`.

    We do make assertions, but they're not verifying that the intended strategies are being
    used. For that we would probably need some additional logging facilities -- possible future
    work.

    For now, this test is largely here just to support "white box" testing, i.e. where you
    want to use the debugger and manually step through and confirm that the expected steps are
    indeed taking place. It also helps to achieve better code coverage, by making some queries
    with the basic heuristic filtering strategy switched off.
    """
    with tempfile.TemporaryDirectory() as dirname:
        # g1 will use basic heuristic filtering strategy, while g2 will not.
        g1, remote1 = get_basic_setup(dirname)
        config2 = GremliteConfig()
        config2.use_basic_heuristic_filtering = False
        g2, remote2 = get_basic_setup(dirname, air_routes=False, config=config2)

        # Here, the `has_label('airport')` should be performed query-side, while
        # the `has('code', TextP.lt('AZZ'))` should be performed element-side.
        result = g1.V().has_label('airport').has('code', TextP.lt('AZZ')).count().next()
        assert result == 2

        # This time, both `has()` steps should be performed query-side, since we are on
        # g2, which is not using the heuristic.
        result = g2.V().has_label('airport').has('code', TextP.ending_with('S')).count().next()
        assert result == 1

        # Here, since we have *only* a text-predicate filter, and the incoming query is
        # "broad", the filter should be applied query-side.
        result = g1.V().has('code', TextP.lt('AZZ')).count().next()
        assert result == 2

        # This time the incoming query is "narrow", so the filter should be applied element-side.
        result = g1.V(1, 3, 5, 7, 9).has('code', TextP.lt('AZZ')).count().next()
        assert result == 2


def test_has_property_at_all():
    """
    Test the filter pattern `has(property_name)`.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Add some properties.

        # A property possessed by all vertices?
        #   They already all have the 'code' property.
        # A property possessed by all edges:
        g.E().property('x', 'y').iterate()

        # A property possessed only by one vertex:
        g.V().has('code', 'AUS').property('foo', 'bar').iterate()
        # A property possessed by only one edge:
        g.E(11).property('foo', 'bar').iterate()

        # A vertex property in higher cardinality:
        g.V().has('code', 'JFK') \
            .property(Cardinality.set_, 'cat', 'black') \
            .property(Cardinality.set_, 'cat', 'orange') \
            .iterate()
        # An edge property in higher cardinality?
        #   No, this actually isn't supported (in TinkerGraph, and hence not here either).

        # V / all
        result = g.V().has('code').count().next()
        assert result == 5
        # E / all
        result = g.E().has('x').count().next()
        assert result == 9

        # V / all, but only interested in certain vertices
        result = g.V(1, 3).has('code').count().next()
        assert result == 2
        # E / all, but only interested in certain edges
        result = g.E(11, 12, 13).has('x').count().next()
        assert result == 3

        # V / one
        result = g.V().has('foo').count().next()
        assert result == 1
        # E / one
        result = g.E().has('foo').count().next()
        assert result == 1

        # V / none (and the name isn't even in the strings table)
        result = g.V().has('spam').count().next()
        assert result == 0
        # E / none (but the name is in the strings table)
        result = g.E().has('code').count().next()
        assert result == 0

        # Here we want to test that properties in plural cardinality do not result in the
        # matched objects being returned more than once.
        result = g.V().has('cat').count().next()
        assert result == 1


@pytest.mark.parametrize(['read_all'], [
    [False],
    [True]
])
def test_V(read_all):
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        config.read_all_at_once = read_all
        g, remote = get_basic_setup(dirname, config=config)

        v = g.V(1).next()
        # You can pass a vertex itself to the V() step:
        result = g.V(v).values('code').next()
        assert result == 'AUS'

        e = g.E(12).next()
        # But you can't pass an edge:
        with pytest.raises(BadStepArgs):
            g.V(e).id_().next()

        # You can pass a mixture of vertices and vertex IDs:
        result = g.V(v, 3).id_().to_set()
        assert result == {1, 3}

        # Passing one list instead:
        result = g.V([v, 3]).id_().to_set()
        assert result == {1, 3}


@pytest.mark.parametrize(['read_all'], [
    [False],
    [True]
])
def test_E(read_all):
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        config.read_all_at_once = read_all
        g, remote = get_basic_setup(dirname, config=config)

        edges = g.E(12, 15, 17).to_list()
        print()
        print(edges)
        assert {e.id for e in edges} == {12, 15, 17}

        # This one uses quad constraint pattern [p|g]
        result = g.E().has_label('route').count().next()
        assert result == 9

        # This one uses quad constraint pattern [pg|]
        result = g.E(12).has_label('route').count().next()
        assert result == 1

        e = g.E(12).next()
        # You can pass an edge itself to the E() step:
        result = g.E(e).id_().next()
        assert result == 12

        # You can pass a mixture of edges and edge IDs:
        result = g.E(e, 13).id_().to_set()
        assert result == {12, 13}

        # Passing one list instead:
        result = g.E([e, 13]).id_().to_set()
        assert result == {12, 13}


def test_standalone_has_steps():
    """
    Many steps incorporate subsequent `has()` and `has_label()` as contributing
    steps, but some do not; furthermore, these steps sometimes have to be initial
    ones, in ongoing traversals `__.has()` etc. So we have to test them as
    standalone steps.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # The in_v() and out_v() steps are ones that produce vertices, but
        # do *not* accept `has` filters as contributing steps. So they provide
        # a way to test.

        result = g.E(11).in_v().values('code').next()
        assert result == 'DFW'

        assert g.E(11).in_v().has('code').has_next() is True
        assert g.E(11).in_v().has('foo').has_next() is False

        assert g.E(11).in_v().has('code', 'DFW').has_next() is True
        assert g.E(11).in_v().has('code', P.within(['AUS', 'DFW', 'FOO'])).has_next() is True
        assert g.E(11).in_v().has('code', TextP.ending_with('FW')).has_next() is True

        # By selecting a finite set of vertices (and using the basic filtering heuristic,
        # which is the default mode), we know that the text predicate filter step will
        # be performed on the element side, i.e. using a standalone `has()` step.
        assert g.V(1, 3, 5, 7, 9).has('code', TextP.gt('LAX')).has_next() is False
        assert g.V(1, 3, 5, 7, 9).has('code', TextP.gte('LAX')).has_next() is True
        assert g.V(1, 3, 5, 7, 9).has('code', TextP.lte('ATL')).has_next() is True
        assert g.V(1, 3, 5, 7, 9).has('code', TextP.lt('ATL')).has_next() is False

        assert g.E(11).in_v().has('code', 'FOO').has_next() is False

        assert g.E(11).in_v().has_label('airport', 'foobar').has_next() is True
        assert g.E(11).in_v().has_label(TextP.starting_with('air')).has_next() is True
        assert g.E(11).in_v().has_label('foobar').has_next() is False

        with pytest.raises(BadStepArgs):
            g.E(11).in_v().has('too', 'many', 'args', 'here').has_next()


def test_select():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().has('airport', 'code', 'AUS').as_('a') \
            .out().has('code', 'DFW').as_('b') \
            .out().has('code', 'JFK').as_('c') \
            .select('a', 'b', 'c').by('code').by().next()
        assert isinstance(result, dict)
        assert list(result.keys()) == ['a', 'b', 'c']
        assert result['a'] == 'AUS'
        assert result['c'] == 'JFK'
        b = result['b']
        assert isinstance(b, graph.Vertex)
        assert b.label == 'airport'
        p = b.properties
        assert len(p) == 1
        assert p[0].key == 'code'
        assert p[0].value == 'DFW'

        # Consider properties with multiple values
        g.V().has('code', 'JFK').property(Cardinality.set_, 'code', 'NYC').iterate()

        # __.properties() traversal selects first property *object*
        result = g.V().has('code', 'JFK').as_('a').select('a').by(__.properties('code')).next()
        print('\n', result)
        assert isinstance(result, graph.VertexProperty)
        assert result.key == 'code' and result.value == 'JFK'

        # Below we differ from TinkerGraph 3.7.2, which will not allow you to use either
        # a __.values() traversal, or a property name, when that property has multiple values.
        # We do allow both, and you simply get the first property *value* in each case.

        result = g.V().has('code', 'JFK').as_('a').select('a').by(__.values('code')).next()
        print('\n', result)
        assert isinstance(result, str)
        assert result == 'JFK'

        result = g.V().has('code', 'JFK').as_('a').select('a').by('code').next()
        print('\n', result)
        assert isinstance(result, str)
        assert result == 'JFK'


def test_path():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Modulating vertices to their 'code' prop, and edges to their ID:
        p = g.V().has('code', 'AUS').as_('start').out_e().as_('hop1') \
            .in_v().has('code', 'DFW').as_('layover').out_e() \
            .in_v().has('code', 'JFK').as_('dest') \
            .path().by(__.properties('code').value()).by(__.id_()) \
            .next()

        assert isinstance(p, graph.Path)
        assert p.objects == ['AUS', 11, 'DFW', 15, 'JFK']
        assert p.labels == [{'start'}, {'hop1'}, {'layover'}, set(), {'dest'}]

        # This time keep vertices as vertices
        p = g.V().has('code', 'AUS').as_('start').out_e().as_('hop1') \
            .in_v().has('code', 'DFW').as_('layover').out_e() \
            .in_v().has('code', 'JFK').as_('dest') \
            .path().by().by(__.id_()) \
            .next()

        assert isinstance(p, graph.Path)
        assert len(p.objects) == 5
        assert isinstance(p.objects[0], graph.Vertex)

        # If any modulator is nonproductive, the whole path fails:
        assert g.V().has('code', 'AUS').as_('start').out_e().as_('hop1') \
            .in_v().has('code', 'DFW').as_('layover').out_e() \
            .in_v().has('code', 'JFK').as_('dest') \
            .path().by('foo').by(__.id_()) \
            .has_next() is False

        # Try giving a single vertex multiple labels, and at different times:
        p = g.V().has('code', 'DFW').as_('d') \
            .V().has('code', 'JFK').as_('j') \
            .V().has('code', 'LAX') \
            .V().has('code', 'DFW').as_('x', 'y').path().by('code').next()
        assert isinstance(p, graph.Path)
        assert p.objects == ['DFW', 'JFK', 'LAX', 'DFW']
        assert p.labels == [{'d'}, {'j'}, set(), {'y', 'x'}]


def test_project():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Give JFK a second code, and a 'state' property.
        g.V().has('code', 'JFK').property(Cardinality.set_, 'code', 'NYC').property('state', 'NY').iterate()

        result = g.V().has('code', 'JFK').project('id', 'c', 's') \
            .by(__.id_()).by(__.values('code').fold()).by('state').next()

        assert result == {'id': 7, 'c': ['JFK', 'NYC'], 's': 'NY'}

        result = g.V().has('code', 'JFK').project('id', 'c', 's', 'vertex') \
            .by(__.id_()).by(__.values('code').fold()).by('state').by().next()

        assert isinstance(result['vertex'], graph.Vertex)

        # If any selector is non-productive, we just omit that key; the rest proceeds.
        result = g.V().has('code', 'JFK').project('id', 'c', 's') \
            .by(__.id_()).by(__.values('code').fold()).by('FOOBAR!').next()

        assert result == {'id': 7, 'c': ['JFK', 'NYC']}

        # This works even if *all* selectors fail.
        result = g.V().has('code', 'JFK').project('id', 'c', 's') \
            .by('foo').by('bar').by('FOOBAR!').next()

        assert result == {}

        # Modulators cycle:
        result = g.V().has('code', 'JFK').project('id', 'c', 's').by(__.id_()).by('foo').next()

        assert result == {'id': 7, 's': 7}


@pytest.mark.parametrize(['read_all'], [
    [False],
    [True]
])
def test_traversal_steps_without_edge_labels(read_all):
    """
    Test all of the in/out steps, without passing any edge label args.
    Here we can use the mini air routes graph.
    """
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        config.read_all_at_once = read_all
        g, remote = get_basic_setup(dirname, config=config)

        # out_e() / in_v()
        L = g.V().has('airport', 'code', 'AUS').out_e().in_v().element_map().to_list()
        assert {e['code'] for e in L} == {'DFW', 'ATL'}

        L = g.V().has('airport', 'code', 'AUS').out_e().has_label('foo').in_v().element_map().to_list()
        assert {e['code'] for e in L} == set()

        # out()
        L = g.V().has('airport', 'code', 'AUS').out().element_map().to_list()
        assert {e['code'] for e in L} == {'DFW', 'ATL'}

        L = g.V().has('airport', 'code', 'AUS').out().has('code', 'ATL').element_map().to_list()
        assert {e['code'] for e in L} == {'ATL'}

        # in_e() / out_v()
        L = g.V().has('airport', 'code', 'AUS').in_e().out_v().element_map().to_list()
        assert {e['code'] for e in L} == {'LAX'}

        # in_()
        L = g.V().has('airport', 'code', 'AUS').in_().element_map().to_list()
        assert {e['code'] for e in L} == {'LAX'}

        # both_v()
        L = g.E(11).both_v().values('code').to_list()
        assert L == ['AUS', 'DFW']

        # other_v()
        L = g.V().has('code', 'AUS') \
            .out_e('route').where(__.in_v().has('code', 'DFW')) \
            .other_v().values('code').to_list()
        assert L == ['DFW']

        L = g.V().has('code', 'DFW') \
            .in_e('route').where(__.out_v().has('code', 'AUS')) \
            .other_v().values('code').to_list()
        assert L == ['AUS']

        # both()
        L = g.V().has('code', 'LAX').in_().values('code').to_list()
        assert L == ['DFW']

        L = g.V().has('code', 'LAX').out().values('code').to_list()
        assert len(L) == 3
        assert set(L) == {'JFK', 'AUS', 'DFW'}

        L = g.V().has('code', 'LAX').both().values('code').to_list()
        assert len(L) == 4
        assert set(L) == {'JFK', 'AUS', 'DFW'}
        assert L.count('DFW') == 2

        # both_e()
        L = g.V().has('code', 'LAX').in_e().id_().to_list()
        assert L == [16]

        L = g.V().has('code', 'LAX').out_e().id_().to_list()
        assert len(L) == 3
        assert set(L) == {17, 18, 19}

        L = g.V().has('code', 'LAX').both_e().id_().to_list()
        assert len(L) == 4
        assert set(L) == {16, 17, 18, 19}


def test_traversal_steps_with_edge_labels():
    """
    Test all of the in/out steps, this time passing various edge label args.
    For this we need to add some other edge types to the mini air routes graph.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Add some stuff to the graph:
        g.add_v('airport').property('code', 'SAT').as_('sat') \
            .V().has('code', 'AUS').add_e('drive').to('sat').iterate()
        remote.commit()

        # Now try filtering edge traversal steps using label names.

        # Don't name any labels. All are accepted.
        result = g.V().has('code', 'AUS').out().values('code').to_set()
        assert result == {'ATL', 'DFW', 'SAT'}

        # Name all labels. Get the same set.
        result = g.V().has('code', 'AUS').out('route', 'drive').values('code').to_set()
        assert result == {'ATL', 'DFW', 'SAT'}

        # Accept only `route` edges.
        result = g.V().has('code', 'AUS').out('route').values('code').to_set()
        assert result == {'ATL', 'DFW'}

        # Accept only `drive` edges.
        result = g.V().has('code', 'AUS').out('drive').values('code').to_set()
        assert result == {'SAT'}


def test_property_value_out_of_range():
    """
    For now, on an out of range integer, we're just returning `None` for its encoding,
    and failing silently. In the future might want to raise an exception instead?
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().has('foo', 2**62).count().next()
        assert result == 0


def test_read_properties():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        g.E().property('time', 3).iterate()

        result = g.V().properties().to_list()
        print()
        print(result)
        assert len(result) == 5
        assert isinstance(result[0], graph.VertexProperty)
        assert {r.key for r in result} == {'code'}
        assert {r.value for r in result} == {'AUS', 'DFW', 'LAX', 'JFK', 'ATL'}

        result = g.V().values().to_list()
        print()
        print(result)
        assert len(result) == 5
        assert isinstance(result[0], str)
        assert set(result) == {'AUS', 'DFW', 'LAX', 'JFK', 'ATL'}

        result = g.E().properties().to_list()
        print()
        print(result)
        assert len(result) == 9
        assert isinstance(result[0], graph.Property)
        assert {r.key for r in result} == {'time'}
        assert {r.value for r in result} == {3}

        result = g.V().properties().key().to_list()
        print()
        print(result)
        assert result == ['code'] * 5

        result = g.V().properties().value().to_list()
        print()
        print(result)
        assert set(result) == {'AUS', 'DFW', 'LAX', 'JFK', 'ATL'}

        result = g.E().properties().key().to_list()
        print()
        print(result)
        assert result == ['time'] * 9

        result = g.E().properties().value().to_list()
        print()
        print(result)
        assert result == [3] * 9


def test_fold_unfold():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().values().fold().next()
        assert set(result) == {'AUS', 'DFW', 'LAX', 'JFK', 'ATL'}

        result = g.V().values().fold().unfold().to_list()
        assert set(result) == {'AUS', 'DFW', 'LAX', 'JFK', 'ATL'}


def test_restart_barriers():
    """
    Test that the "restart barrier" steps produce a path of length
    one, containing only their evaluation of the incoming stuff.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # With no barrier, we're getting paths of length three:
        result = g.V().out().out().path().next()
        assert len(result.objects) == 3

        # The restart barriers should reduce to a path of length 1.

        # count():
        result = g.V().out().out().count().path().next()
        assert len(result.objects) == 1

        # fold():
        result = g.V().out().out().fold().path().next()
        assert len(result.objects) == 1


def test_limit():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().values().to_list()
        print('\n', result)
        assert len(result) == 5

        result = g.V().values().limit(3).to_list()
        print('\n', result)
        assert len(result) == 3

        result = g.V().values().limit(10).to_list()
        print('\n', result)
        assert len(result) == 5

        result = g.V().values().limit(0).to_list()
        print('\n', result)
        assert len(result) == 0


def test_property_types():
    """
    Try setting and getting properties of different types.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        g.V(1) \
            .property('a', None) \
            .property('b', True) \
            .property('c', False) \
            .property('d', 7) \
            .property('e', 3.14) \
            .property('f', 'foo') \
            .iterate()

        em = g.V(1).element_map().next()
        for name, type_ in zip('abcdef', (type(None), bool, bool, int, float, str)):
            assert isinstance(em[name], type_)


def test_set_properties():
    """
    Try setting vertex and edge properties, with different cardinalities.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # vertex / single
        # Check initial value:
        P = g.V(1).properties('code').to_list()
        assert [p.value for p in P] == ['AUS']
        # Default cardinality is single. So if we set a new value,
        # it replaces the old one:
        g.V(1).property('code', 'FOO').iterate()
        P = g.V(1).properties('code').to_list()
        assert [p.value for p in P] == ['FOO']

        # vertex / set
        # Add a new value, under "set" card.:
        g.V(1).property(Cardinality.set_, 'code', 'BAR').iterate()
        P = g.V(1).properties('code').to_list()
        L = [p.value for p in P]
        assert len(L) == 2
        assert L.count("FOO") == 1
        assert L.count("BAR") == 1
        # If we do the same thing again, we haven't changed the set:
        g.V(1).property(Cardinality.set_, 'code', 'BAR').iterate()
        P = g.V(1).properties('code').to_list()
        L = [p.value for p in P]
        assert len(L) == 2
        assert L.count("FOO") == 1
        assert L.count("BAR") == 1

        # vertex / list
        # If we add an existing value, but now under "list" card., it extends the list:
        g.V(1).property(Cardinality.list_, 'code', 'BAR').iterate()
        P = g.V(1).properties('code').to_list()
        L = [p.value for p in P]
        assert len(L) == 3
        assert L.count("FOO") == 1
        assert L.count("BAR") == 2

        # edge / list
        with pytest.raises(UnsupportedUsage):
            g.E(11).property(Cardinality.list_, 'foo', 'bar').iterate()

        # edge / set
        with pytest.raises(UnsupportedUsage):
            g.E(11).property(Cardinality.set_, 'foo', 'bar').iterate()

        # edge / single
        # Set a value:
        g.E(11).property('time', 1).iterate()
        P = g.E(11).properties('time').to_list()
        assert [p.value for p in P] == [1]
        # If we set it again, we replace the existing value:
        g.E(11).property('time', 2).iterate()
        P = g.E(11).properties('time').to_list()
        assert [p.value for p in P] == [2]


def test_repeat():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        AUS, ATL, DFW, LAX, JFK = "AUS ATL DFW LAX JFK".split()

        # Add an extra edge, so that more loops are possible.
        g.V().has('code', 'JFK').as_('j').V().has('code', 'ATL').add_e('route').from_('j').iterate()

        # until(), but no emit(): Every path should end with JFK.
        result = g.V(1).repeat(__.out()).until(__.has('code', 'JFK')).path().by('code').limit(10).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        assert all(p[-1] == JFK for p in L)

        # emit(), but no until(): Paths can end with anything, and can go on beyond JFK.
        result = g.V(1).repeat(__.out()).emit().path().by('code').limit(10).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        assert any(JFK in p and p[-1] != JFK for p in L)

        # both emit() and until(): Paths can end with anything, but if they ever reach JFK, then they do not go beyond.
        result = g.V(1).repeat(__.out()).emit().until(__.has('code', 'JFK')).path().by('code').limit(10).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        assert any(p[-1] != JFK for p in L)
        assert any(p[-1] == JFK for p in L)
        assert all(p[-1] == JFK or JFK not in p for p in L)

        # Use `times()`
        result = g.V(1).repeat(__.out()).emit().times(2).path().by('code').to_list()
        L = paths_to_lists(result)
        print('\n', L)
        expected = [['AUS', 'DFW'], ['AUS', 'ATL'], ['AUS', 'DFW', 'LAX'], ['AUS', 'DFW', 'JFK'],
                          ['AUS', 'ATL', 'DFW'], ['AUS', 'ATL', 'JFK']]
        assert set(tuple(p) for p in L) == set(tuple(p) for p in expected)

        # Put `emit()` *before* `repeat()`
        result = g.V(1).emit().repeat(__.out()).times(2).path().by('code').to_list()
        L = paths_to_lists(result)
        print('\n', L)
        assert L[0] == [AUS]

        # Put `until()` *before* `repeat()`
        # This time we start at LAX, and ask for paths up until reaching LAX.
        # If the `until()` comes after the `repeat()`, then we find many paths, because we have already
        # moved away from LAX before we start asking if we have reached it:
        result = g.V().has('code', 'LAX') \
            .repeat(__.out()).until(__.has('code', 'LAX')) \
            .path().by('code').limit(5).to_list()
        assert len(result) == 5
        # But if the `until()` comes before the `repeat()`, then we only produce the one, one-element path,
        # because every path already begins at a point that meets the halting condition.
        result = g.V().has('code', 'LAX') \
            .until(__.has('code', 'LAX')).repeat(__.out()) \
            .path().by('code').limit(5).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        assert L == [[LAX]]


def test_as():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Apply multiple labels at once:
        result = g.V().has('code', 'AUS').as_('a', 'b', 'c') \
            .select('a', 'b', 'c').by('code').by(__.id_()).by(__.label()).next()
        assert result == {'a': 'AUS', 'b': 1, 'c': 'airport'}

        # Filtering inside a `where()` step:

        # First, without any filtering:
        result = g.V().has('code', 'DFW').as_('d').V().has('code', 'JFK').in_() \
            .where(__.identity()).values('code').to_set()
        assert result == {'LAX', 'ATL', 'DFW'}

        # Now we add an `as_()` inside the `where()`, which should act as a filter, requiring that
        # the objects equal the one of the given label.
        result = g.V().has('code', 'DFW').as_('d').V().has('code', 'JFK').in_() \
            .where(__.identity().as_('d')).values('code').to_list()
        assert result == ['DFW']


def test_multi_repeat():
    """
    Test that a traversal with multiple repeat() steps parses correctly (and works!).
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname, air_routes=False)

        # Build this graph:
        #
        #           X
        #            \
        #             v
        #    A ------> B ------> C
        #             ^
        #            /
        #    Z ---> Y
        #
        g.add_v('foo').property('name', 'A').as_('a') \
            .add_v('foo').property('name', 'B').as_('b') \
            .add_v('foo').property('name', 'C').as_('c') \
            .add_v('foo').property('name', 'X').as_('x') \
            .add_v('foo').property('name', 'Y').as_('y') \
            .add_v('foo').property('name', 'Z').as_('z') \
            .add_e('bar').from_('a').to('b') \
            .add_e('bar').from_('b').to('c') \
            .add_e('bar').from_('x').to('b') \
            .add_e('bar').from_('y').to('b') \
            .add_e('bar').from_('z').to('y') \
            .iterate()

        # .repeat().emit().repeat().emit()
        result = g.V().has('name', 'A') \
            .repeat(__.out()).emit().repeat(__.in_()).emit() \
            .path().by('name').limit(12).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        expected = [['A', 'B', 'A'], ['A', 'B', 'X'], ['A', 'B', 'Y'], ['A', 'B', 'Y', 'Z'],
                    ['A', 'B', 'C', 'B'], ['A', 'B', 'C', 'B', 'A'], ['A', 'B', 'C', 'B', 'X'],
                    ['A', 'B', 'C', 'B', 'Y'], ['A', 'B', 'C', 'B', 'Y', 'Z']]
        assert set(tuple(p) for p in L) == set(tuple(p) for p in expected)

        # .emit().repeat().emit().repeat()
        result = g.V().has('name', 'A') \
            .emit().repeat(__.out()).emit().repeat(__.in_()) \
            .path().by('name').limit(12).to_list()
        L = paths_to_lists(result)
        print('\n', L)
        expected = [['A'], ['A', 'B'], ['A', 'B', 'A'], ['A', 'B', 'X'], ['A', 'B', 'Y'], ['A', 'B', 'Y', 'Z'],
                    ['A', 'B', 'C'], ['A', 'B', 'C', 'B'], ['A', 'B', 'C', 'B', 'A'], ['A', 'B', 'C', 'B', 'X'],
                    ['A', 'B', 'C', 'B', 'Y'], ['A', 'B', 'C', 'B', 'Y', 'Z']]
        assert set(tuple(p) for p in L) == set(tuple(p) for p in expected)


def test_store_cap():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        AUS, ATL, DFW, LAX, JFK = "AUS ATL DFW LAX JFK".split()

        def check_list(codes):
            print('\n', codes)
            assert len(codes) == 6
            assert set(codes) == {DFW, ATL, JFK, LAX}
            assert codes.count(DFW) == 2
            assert codes.count(JFK) == 2

        # We first confirm that we get expected results by following paths.
        # Try moving outward twice from AUS.
        result = g.V(1).repeat(__.out()).emit().times(2).path().by('code').to_list()
        L = paths_to_lists(result)
        # We expect to find these paths:
        # [['AUS', 'DFW'], ['AUS', 'ATL'], ['AUS', 'DFW', 'LAX'], ['AUS', 'DFW', 'JFK'],
        #  ['AUS', 'ATL', 'DFW'], ['AUS', 'ATL', 'JFK']]
        # So if we take the final code in each path...
        finals = [path[-1] for path in L]
        # We expect ['DFW', 'DFW', 'ATL', 'JFK', 'JFK', 'LAX'] in some order.
        check_list(finals)

        # Now, instead of returning paths, we use store and cap to get the list of codes reached.
        codes_reached = g.V(1).repeat(__.out().store('codes').by('code')).times(2).cap('codes').next()
        # It should be the same as the `finals` list from above (in some order).
        check_list(codes_reached)


def test_union():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().has('code', TextP.starting_with('A')).values('code').to_set()
        assert result == {'ATL', 'AUS'}

        result = g.V().has('code', TextP.containing('A')).values('code').to_set()
        assert result == {'ATL', 'LAX', 'AUS'}

        result = g.V().union(
            __.has('code', TextP.starting_with('A')),
            __.has('code', TextP.containing('A'))
        ).values('code').to_set()
        assert result == {'ATL', 'LAX', 'AUS'}

        # This time, instead of asking for a set, we ask for a list. This is so we can
        # verify that repeats are *not* eliminated.
        # However, I am not trying to replicate the *ordering* of the list that we get in TinkerGraph,
        # because I do not understand it. In TinkerGraph, you get:
        #   ['AUS', 'AUS', 'LAX', 'ATL', 'ATL']
        # What is the ordering? It is not alphabetical, and it is not the order in which the results
        # were encountered in the constituent traversals.
        result = g.V().union(
            __.has('code', TextP.starting_with('A')),
            __.has('code', TextP.containing('A'))
        ).values('code').to_list()
        assert len(result) == 5
        assert result.count('AUS') == 2
        assert result.count('LAX') == 1
        assert result.count('ATL') == 2

        # Show that it's okay to union just one traversal:
        result = g.V().union(__.has('code', TextP.starting_with('A'))).values('code').to_set()
        assert result == {'ATL', 'AUS'}

        # Show that it's okay to union *no* traversals:
        result = g.V().union().values('code').has_next()
        assert result is False


def test_and_or_where_not():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # or() with two traversals
        result = g.V().or_(
            __.has('code', TextP.starting_with('A')),
            __.has('code', TextP.starting_with('L'))
        ).values('code').to_set()
        assert result == {'ATL', 'LAX', 'AUS'}

        # or() with one traversal
        result = g.V().or_(__.has('code', TextP.starting_with('A'))).values('code').to_set()
        assert result == {'ATL', 'AUS'}

        # and() with two traversals
        result = g.V().and_(__.has('code', TextP.containing('A')), __.has('code', TextP.containing('L'))).values('code').to_set()
        assert result == {'ATL', 'LAX'}

        # and() with one traversal
        result = g.V().and_(__.has('code', TextP.containing('A'))).values('code').to_set()
        assert result == {'ATL', 'LAX', 'AUS'}

        # where()
        result = g.V().where(__.has('code', TextP.containing('A'))).values('code').to_set()
        assert result == {'ATL', 'LAX', 'AUS'}

        # not()
        result = g.V().not_(
            __.and_(
                __.has('code', TextP.containing('A')),
                __.has('code', TextP.containing('L'))
            )
        ).values('code').to_set()
        assert result == {'JFK', 'AUS', 'DFW'}


def test_text_order_constraints():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().has('code', TextP.lt('JFK')).values('code').to_set()
        assert result == {'ATL', 'AUS', 'DFW'}

        result = g.V().has('code', TextP.lte('JFK')).values('code').to_set()
        assert result == {'ATL', 'JFK', 'AUS', 'DFW'}

        result = g.V().has('code', TextP.gte('JFK')).values('code').to_set()
        assert result == {'LAX', 'JFK'}

        result = g.V().has('code', TextP.gt('JFK')).values('code').to_set()
        assert result == {'LAX'}


def test_predicates():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().has('code', P.within([
            'AUS', 'JFK', 'FOO', True, 3.14, 7, None
        ])).values('code').to_set()
        assert result == {'JFK', 'AUS'}

        result = g.V().has_label(P.within([
            'airport', 'foobar'
        ])).count().next()
        assert result == 5

        result = g.E().has_label(P.within(['route', 'foobar'])).count().next()
        assert result == 9

        result = g.V().has_label(TextP.starting_with('airp')).count().next()
        assert result == 5

        result = g.E().has_label(TextP.starting_with('rou')).count().next()
        assert result == 9


def test_barrier():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Working *without* a `barrier()` step.
        # We locate the 'AUS' vertex, and then we do a union over `identity()` and `out()`, which will eventually
        # iterate over 'AUS', 'DFW', and 'ATL'. If we add an outgoing 'route' edge to 'LAX' *without* first putting
        # a barrier step, then we're going to add an edge AUS --> LAX *before* the `union` has carried out the
        # `out` traversal. Therefore, when it does carry that out, it will find that LAX is now among the reachable
        # vertices. This means then, when we're all done, LAX is going to have a self-loop.
        g.V().has('code', 'LAX').as_('lax').V().has('code', 'AUS').union(
            __.identity(), __.out('route')
        ).add_e('route').to('lax').iterate()
        # LAX has a self-loop:
        assert g.E().and_(__.out_v().has('code', 'LAX'), __.in_v().has('code', 'LAX')).has_next() is True

        # Get a fresh database
        g, remote = get_basic_setup(dirname, filename='bar.db')

        # Working *with* a `barrier()` step.
        # This time, we do put a `barrier()` step right after the `union`, and before the `add_e`.
        # This means that the union will first locate all the nodes of interest, AUS, DWF, and ATL, and
        # only *then* will we begin adding edges. This time, LAX will *not* wind up with a self-loop.
        g.V().has('code', 'LAX').as_('lax').V().has('code', 'AUS').union(
            __.identity(), __.out('route')
        ).barrier().add_e('route').to('lax').iterate()
        # LAX does *not* have a self-loop:
        assert g.E().and_(__.out_v().has('code', 'LAX'), __.in_v().has('code', 'LAX')).has_next() is False


def test_order():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # ascending
        result = g.V().order().by('code').values('code').to_list()
        assert result == ['ATL', 'AUS', 'DFW', 'JFK', 'LAX']

        # descending
        result = g.V().order().by('code', Order.desc).values('code').to_list()
        assert result == ['LAX', 'JFK', 'DFW', 'AUS', 'ATL']

        # shuffle
        results = set()
        for k in range(20):
            result = g.V().order().by('code', Order.shuffle).values('code').to_list()
            results.add(tuple(result))
            if len(results) > 1:
                break
        else:
            # 20 random sortings of a list that has 120 possible orderings were all the same?
            # That's a 1 in 120^19 chance. Something is wrong.
            assert False

        ########################################################
        # Sorting graph objects themselves

        # Vertices, Edges, and VertexProperties are by ID:
        result = g.V().order().id_().to_list()
        assert result == [1, 3, 5, 7, 9]

        result = g.E().order().id_().to_list()
        assert result == list(range(11, 20))

        result = g.V().properties('code').order().id_().to_list()
        assert result == [2, 4, 6, 8, 10]

        # Edge properties are by value:
        g.V().has('code', 'AUS').out_e().property('time', 3).iterate()
        g.V().has('code', 'LAX').out_e().property('time', 5).iterate()
        result = g.E().properties().order().value().to_list()
        assert result == [3, 3, 5, 5, 5]

        ########################################################

        # Add more properties, so we can try a primary / secondary sort
        g.V().has('code', 'ATL').property('tz', 'Eastern') \
            .V().has('code', 'AUS').property('tz', 'Central') \
            .V().has('code', 'DFW').property('tz', 'Central') \
            .V().has('code', 'JFK').property('tz', 'Eastern') \
            .V().has('code', 'LAX').property('tz', 'Pacific') \
            .iterate()

        # Sort by time zome asc first, then by code desc:
        result = g.V().order().by('tz').by('code', Order.desc).project('tz', 'code').by('tz').by('code').to_list()
        assert result == [
            {'tz': 'Central', 'code': 'DFW'}, {'tz': 'Central', 'code': 'AUS'},
            {'tz': 'Eastern', 'code': 'JFK'}, {'tz': 'Eastern', 'code': 'ATL'},
            {'tz': 'Pacific', 'code': 'LAX'}
        ]

        ########################################################
        # Non-productive by() traversals

        # In TinkerGraph 3.7.2, if you request an ordering traversal, then any objects on which
        # that traversal is non-productive are filtered out. Surprisingly, this remains true
        # *even if* that traversal is a secondary or later key, and turns out not to be necessary
        # in order to resolve any orderings. And this remains true even if there is an Order.shuffle
        # present anywhere.

        # Primary key is a property that no object has:
        result = g.V().order().by('foobar').by('code').values('code').to_list()
        assert result == []

        # Secondary key is a property that no object has.
        # Even though the primary key is completely determinative, we still get an empty list.
        result = g.V().order().by('code').by('foobar').values('code').to_list()
        assert result == []

        # Even shuffling doesn't skip the filtering.
        result = g.V().order().by('code', Order.shuffle).by('foobar').values('code').to_list()
        assert result == []


def test_side_effect():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Here we move to the outgoing edges from AUS, store their target endpoints
        # in a global list, but then filter down to nothing because we look for label 'route',
        # which the edges have, but the target vertices to which we have moved do not.
        result = g.V().has('code', 'AUS').out_e() \
            .in_v().store('targets').by('code') \
            .where(__.has_label('route')) \
            .has_next()
        assert result is False

        # This time we instead do the storage in a side effect.
        # This makes it so that we remain at the edges, and filtering on label 'route'
        # does not eliminate anything.
        result = g.V().has('code', 'AUS').out_e() \
            .side_effect(__.in_v().store('targets').by('code')) \
            .where(__.has_label('route')) \
            .has_next()
        print('\n', result)
        assert result is True

        # Check that we do build the expected storage list:
        result = g.V().has('code', 'AUS').out_e() \
            .side_effect(__.in_v().store('targets').by('code')) \
            .where(__.has_label('route')) \
            .cap('targets').next()
        print('\n', result)
        assert len(result) == 2
        assert set(result) == {'DFW', 'ATL'}


def test_flat_map():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Suppose you want the lists of airports you can reach from
        # each given one. If you try the following...
        result = g.V().out().values('code').fold().to_list()
        # ...you are disappointed, because you just get a list containing a single, flat list:
        assert sorted(result[0]) == sorted(
            ['DFW', 'DFW', 'DFW', 'ATL', 'JFK', 'JFK', 'JFK', 'LAX', 'AUS']
        )

        # Instead, you need a way to group steps together.
        # We can use `flat_map()` for that:
        result = g.V().flat_map(__.out().values('code').fold()).to_list()
        assert set(tuple(sorted(L)) for L in result) == set(tuple(sorted(L)) for L in [
            ['DFW', 'ATL'], ['JFK', 'LAX'], ['JFK', 'AUS', 'DFW'], [], ['DFW', 'JFK']
        ])


def test_interp_query():
    q = qt.InterpolatedQuery('foo', ['cat'])
    r = qt.InterpolatedQuery('bar', ['baz'])

    s = q + r
    assert s.value == 'foobar'
    assert s.params == ['cat', 'baz']

    t = q + 'bar'
    assert t.value == 'foobar'
    assert t.params == ['cat']

    u = 'foo' + r
    assert u.value == 'foobar'
    assert u.params == ['baz']


def test_query_join():
    q = qt.InterpolatedQuery('foo', ['cat'])
    r = qt.InterpolatedQuery('bar', ['baz'])
    s = 'spam'

    j = qt.query_join(' AND ', [])
    assert j == ''

    j = qt.query_join(' AND ', [q, r, s])
    assert j.value == 'foo AND bar AND spam'
    assert j.params == ['cat', 'baz']


def test_prop_with_apostrophe():
    """
    Check that we can successfully run traversals in which strings have apostrophes in them.
    This is verifying that we are using proper string interpolation in our SQLite queries (i.e.
    we're not allowing an apostrophe to prematurely terminate a string).
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        result = g.V().out("can't").to_list()
        assert result == []

        g.V().has('code', 'AUS').property("isn't", 'Aussie').iterate()
        result = g.V().has("isn't").count().next()
        assert result == 1

        result = g.V().has_label("isn't", "wasn't").count().next()
        assert result == 0

        result = g.V().has('code', TextP.lt("can't")).count().next()
        assert result == 5

        result = g.V().has('code', P.within(["isn't", "wasn't"])).count().next()
        assert result == 0


def test_simple_path():
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Vertices are not allowed to be repeated.
        # Here we travel from DFW to LAX, and then we're not allowed to travel back to DFW.
        result = g.V().has('code', 'DFW').out().has('code', 'LAX').out().simple_path().count().next()
        assert result == 2

        # Edges are not allowed to be repeated.
        # Here we travel from AUS to DFW, then ask for incoming edges. DFW has three
        # incoming edges, but the `simple_path()` excludes the one from AUS along
        # which we arrived.
        result = g.V().has('code', 'AUS').out_e().in_v().has('code', 'DFW') \
            .in_e().simple_path().count().next()
        assert result == 2

        # This shows that the `simple_path()` step does not have to modify a step
        # that would have otherwise caused a repeat. Here we visit a vertex twice, and then
        # visit its outgoing edges. None of those edges would have been a repeat, but
        # `simple_path()` still eliminates all results, since they all involve a repeated vertex.
        result = g.V().has('code', 'AUS').V().has('code', 'AUS').out_e().simple_path().count().next()
        assert result == 0


def test_push_pop_result():
    """
    Test pushing and popping of `Result` state, in traversals that involve following
    continuations via `carry_on()`.
    """
    with tempfile.TemporaryDirectory() as dirname:
        g, remote = get_basic_setup(dirname)

        # Here we use a `union()`. This is a step that follows continuations *to exhaustion*.
        # This means that `carry_on()` will *not* invoke the `abort()` method. Following
        # the continuations to exhaustion means that all necessary pop's get a chance to
        # happen, and the `Result` is naturally restored to its incoming state.
        result = g.V(1).out().union(__.id_(), __.values('code')).to_list()
        print('\n', result)
        assert result == [3, 'DFW', 9, 'ATL']

        # Next we examine some traversals in which a `where()` step has us following a
        # continuation that is *not* exhausted. Instead, `where()` only wants to know if
        # the continuation produces one or more results. Because of this, `carry_on()`
        # should invoke the `abort()` method, which gives all necessary pop's a chance to
        # happen, so that again the `Result` is restored to its incoming state.

        # Suppose we want to get the set of out-neighbors of ATL which are not dead-ends,
        # i.e. which also have at least one out-neighbor of their own.
        # ATL has two out-nbrs, DFW and JFK, and of these JFK is a dead end, so we should
        # get just DFW.
        result = g.V().has('code', 'ATL').out().where(__.out()).values('code').to_list()
        print('\n', result)
        assert result == ['DFW']

        # This time we want out-neighbors of ATL from which it is possible to move on
        # not just once but twice. (This again includes DFW, since you can go
        # DFW --> LAX --> JFK for example.) In this example, since the continuation
        # `__.out().out()` actually has more than one step, we get to see that indeed
        # *every* step in the continuation chain gets a chance to call `pop_state()`.
        result = g.V().has('code', 'ATL').out().where(__.out().out()).values('code').to_list()
        print('\n', result)
        assert result == ['DFW']


def test_locked_database():
    """
    Examine circumstances under which the database does or does not become locked,
    involving different connections reading and writing.

    See https://stackoverflow.com/a/53470118
        https://stackoverflow.com/a/53470179
    """
    Nv, Ne = 50, 70
    with tempfile.TemporaryDirectory() as dirname:
        # Form two connections to a single database.
        g1, remote1 = get_basic_setup(dirname, air_routes=False)
        g2, remote2 = get_basic_setup(dirname, air_routes=False, timeout=0.05)
        assert remote2.con is not remote1.con

        def start_reading(remote):
            c = remote.con.cursor()
            c.execute('SELECT * FROM quads')
            return c

        # Scenario 1
        # The first connection can write, even while it has an open reading cursor.
        # Build using conn 1:
        make_random_graph(g1, Nv, Ne)
        remote1.commit()
        # Conn 2 sees the built graph:
        assert g2.V().count().next() == Nv
        # Conn 1 begins reading:
        cur = start_reading(remote1)
        # Conn 1 can drop the graph:
        g1.V().drop().iterate()
        remote1.commit()
        # Conn 2 sees that the graph has been dropped:
        assert g2.V().count().next() == 0
        # Only now do we close the reading cursor.
        cur.close()

        # Scenario 2A
        # If the first connection has an open reading cursor, then the second connection
        # *cannot* write.
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
        with pytest.raises(sqlite3.OperationalError) as e:
            remote2.commit()
        assert str(e.value) == 'database is locked'

        # Scenario 2B
        # Once conn 1 closes its reading cursor, then conn 2 can write.
        cur.close()
        remote2.commit()
        # Conn 1 sees that the graph has been dropped:
        assert g1.V().count().next() == 0


def test_catch_unsupported_pattern():
    """
    Here we make a special config setting that makes it possible for quad
    constraint pattern [p|s] to be used in a query. We do this to verify that
    we will indeed raise an exception in response to this unsupported pattern.
    """
    with tempfile.TemporaryDirectory() as dirname:
        config = GremliteConfig()
        config.use_p_bar_s_pattern = True
        g, remote = get_basic_setup(dirname, config=config)

        with pytest.raises(UnexpectedQuadConstraintPattern):
            g.V().has('code').iterate()


########################################################################################
# FINAL TEST BELOW. ADD NEW TESTS ABOVE THIS POINT.
########################################################################################


def test_qqc_pattern_and_index_coverage():
    """
    Check that all expected quad query constraint patterns were used, and that
    all expected indexes were used.

    *** NOTE: THIS TEST MUST COME LAST! It may fail if all the above tests have not run first, since
        it is scanning the pytest.log generated by running those tests.

    Note: In the future it might be good to perform a stronger check, that *for each* QQC pattern,
    the expected index is used (as recorded in the `qt.expected_quad_constraint_patterns` dictionary).

    I think the way to do this would be to make our `LoggingCursor` class essentially parse the SQL it
    receives, and extract from that the QQC pattern(s) that were being used in each query (if any).
    It could then compare these against what comes back from its "EXPLAIN QUERY PLAN" query. The "parsing"
    could probably amount to some primitive scanning, nowhere near fully supporting all of SQL.

    You might think it could be done more simply by the `cull_log()` function that we are already
    using here, but I don't think it's so easy. Consider a case like

        g.V().has('airport', 'code', 'AUS')

    as occurs for example in our `test_simple_evaluators()` test, above. This results in a query like

        SELECT s FROM quads WHERE s IN (SELECT s FROM quads WHERE s > 0 AND p = 0 AND o = 4) AND p = 5 AND o = 6

    which employs two different constraint patterns: [po|s] in the nested SELECT, and [spo|] in the outer one.
    But the relevant section of the generated pytest.log looks like this:

        QUAD QUERY CONSTRAINTS: [po|s] (querytools.py:806)
        SELECT id FROM strings WHERE v = ? (connection.py:202)
        [(2, 0, 0, 'SEARCH strings USING COVERING INDEX sqlite_autoindex_strings_1 (v=?)')] (connection.py:203)
        SELECT id FROM strings WHERE v = ? (connection.py:202)
        [(2, 0, 0, 'SEARCH strings USING COVERING INDEX sqlite_autoindex_strings_1 (v=?)')] (connection.py:203)
        QUAD QUERY CONSTRAINTS: [spo|] (querytools.py:806)
        SELECT s FROM quads WHERE s IN (SELECT s FROM quads WHERE s > 0 AND p = 0 AND o = 4) AND p = 5 AND o = 6 (connection.py:202)
        [(2, 0, 0, 'SEARCH quads USING COVERING INDEX ops (o=? AND p=? AND s=?)'), (8, 0, 0, 'LIST SUBQUERY 1'), (10, 8, 0, 'SEARCH quads USING COVERING INDEX ops (o=? AND p=? AND s>?)')] (connection.py:203)

    and I note that the two "QUAD QUERY CONSTRAINTS" lines are separated from the query and plan lines (last two above)
    by some intervening junk. So it is not so easy to associate the patterns with the plans, just by scanning the
    lines of the log.
    """
    indices_used, quad_query_constraints_used = cull_log()
    assert indices_used == expected_index_names
    assert quad_query_constraints_used == set(qt.expected_quad_constraint_patterns.keys())

########################################################################################
# DO NOT ADD TESTS HERE. ADD THEM BEFORE THE test_qqc_pattern_and_index_coverage() ABOVE
########################################################################################
