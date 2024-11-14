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

from enum import Enum
import logging
import sqlite3

import gremlin_python.process.traversal as predicates

from .cursors import DummyCursor
from .errors import UnexpectedQuadConstraintPattern


logger = logging.getLogger(__name__)


class Constants:
    MIN_INT = -2**62
    MAX_INT = 2**62 - 1
    HAS_LABEL_PREDICATE = 0
    IMPOSSIBLE_SUBJECT = 0
    NON_ID_G_VALUE = 0


class SubjectTypes(Enum):
    VERTICES = 'VERTICES'
    EDGES = 'EDGES'


class Sign(Enum):
    NEG = -1
    POS = 1

    @classmethod
    def valence(cls, st: SubjectTypes):
        return cls.NEG if st is SubjectTypes.EDGES else cls.POS


def take_next_graphelt_id(con: sqlite3.Connection):
    cur = con.cursor()
    next_id = cur.execute("INSERT INTO graphelts VALUES (NULL) RETURNING id").fetchone()[0]
    cur.close()
    return next_id


def create_vertex(con: sqlite3.Connection, vertex_label: str):
    """
    Fully create a new vertex. Affects both the graphelts table and the quads table.

    :param con: connection
    :param vertex_label: the label for the new vertex
    :return: the ID assigned to the new vertex
    """
    vertex_id = take_next_graphelt_id(con)
    label_id = merge_string(con, vertex_label)
    cur = con.cursor()
    cur.execute("INSERT INTO quads VALUES (?, ?, ?, ?)", [
        vertex_id, Constants.HAS_LABEL_PREDICATE, label_id, Constants.NON_ID_G_VALUE
    ])
    cur.close()
    return vertex_id


def create_edge(con: sqlite3.Connection, source_id: int, edge_label: str, target_id: int):
    """
    Fully create a new edge. Affects both the graphelts table and the quads table.

    :param con: connection
    :param source_id: id of source vertex
    :param edge_label: the label for the new edge
    :param target_id: id of target vertex
    :return: the ID assigned to the new edge
    """
    edge_id = take_next_graphelt_id(con)
    label_id = merge_string(con, edge_label)
    cur = con.cursor()
    cur.execute("INSERT INTO quads VALUES (?, ?, ?, ?)", [source_id, -label_id, target_id, -edge_id])
    cur.close()
    return edge_id


def create_vertex_property(con: sqlite3.Connection, vertex_id: int, prop_name: str, prop_value):
    """
    Fully create a new vertex property. Affects both the graphelts table and the quads table.

    :param con: connection
    :param vertex_id: the id of the vertex
    :param prop_name: the name of the property
    :param prop_value: the value of the property
    :return: the ID assigned to the new vertex property
    """
    prop_id = take_next_graphelt_id(con)
    name_id = merge_string(con, prop_name)
    value_code = encode_property_value(con, prop_value, merge=True)
    cur = con.cursor()
    cur.execute("INSERT INTO quads VALUES (?, ?, ?, ?)", [
        vertex_id, name_id, value_code, prop_id
    ])
    cur.close()
    return prop_id


def record_edge_property(con: sqlite3.Connection, edge_id: int, prop_name: str, prop_value):
    """
    Record an edge property in the quads table. Since there are no edge property graph elements,
    we do not form a new element ID.

    :param con: connection
    :param edge_id: the id of the edge
    :param prop_name: the name of the property
    :param prop_value: the value of the property
    :return: nothing
    """
    name_id = merge_string(con, prop_name)
    value_code = encode_property_value(con, prop_value, merge=True)
    cur = con.cursor()
    cur.execute("INSERT INTO quads VALUES (?, ?, ?, ?)", [
        -edge_id, name_id, value_code, Constants.NON_ID_G_VALUE
    ])
    cur.close()


def replace_vertex_property(con: sqlite3.Connection, old_prop_id, new_prop_value):
    """
    Replace an existing vertex property.

    Removes the existing vertex property as a graph element,
    forms a new graph element, and updates the existing quad row to
    record the new property value and ID.

    :param con: connection
    :param old_prop_id: the existing property ID
    :param new_prop_value: the new value for the property
    :return: nothing
    """
    delete_elements_from_graphelts_table(con, old_prop_id)
    new_prop_id = take_next_graphelt_id(con)
    value_code = encode_property_value(con, new_prop_value, merge=True)
    cur = con.cursor()
    log_qqc('g')
    cur.execute("UPDATE quads SET o = ?, g = ? WHERE g = ?", [value_code, new_prop_id, old_prop_id])
    cur.close()


def update_edge_property(con: sqlite3.Connection, edge_id, prop_name, new_prop_value):
    """
    Update the quads table for a new value of an existing edge property.

    :param con: connection
    :param edge_id: the edge ID
    :param prop_name: the name of the property
    :param new_prop_value: the new value for the property
    :return:
    """
    name_id = string_to_id(con, prop_name)
    value_code = encode_property_value(con, new_prop_value, merge=True)
    cur = con.cursor()
    log_qqc('sp')
    cur.execute("UPDATE quads SET o = ? WHERE s = ? AND p = ?", [value_code, -edge_id, name_id])
    cur.close()


def delete_elements_from_graphelts_table(con: sqlite3.Connection, element_ids):
    """
    Delete one or more elements from the graphelts table.

    This does NOT perform all the necessary actions to completely remove an element
    from the graph. E.g. removing a vertex also requires removing all of its properties
    and connected edges.

    Instead, this is just one step in a complete removal of an element.

    :param con: connection
    :param element_ids: either a single int, or a list of ints
    :return: number of IDs actually deleted
    """
    if isinstance(element_ids, int):
        element_ids = [element_ids]

    id_set = ','.join(str(n) for n in element_ids)
    cur = con.cursor()
    ids_deleted = cur.execute(
        f"DELETE FROM graphelts WHERE id IN ({id_set}) RETURNING id"
    ).fetchall()
    cur.close()

    return len(ids_deleted)


def get_table_stats(con: sqlite3.Connection):
    """
    Get stats on what is in the database.

    :param con: connection
    :return: RowCount
    """
    cur = con.cursor()

    # Check how many we have of each type of object:
    # Vertices:
    vx = cur.execute('SELECT count(*) FROM quads WHERE p = 0').fetchone()[0]
    # Edges:
    eg = cur.execute('SELECT count(*) FROM quads WHERE g < 0').fetchone()[0]
    # Vertex properties:
    vp = cur.execute('SELECT count(*) FROM quads WHERE g > 0').fetchone()[0]
    # Edge properties:
    ep = cur.execute('SELECT count(*) FROM quads WHERE s < 0').fetchone()[0]

    # Sanity checks:
    num_quads = cur.execute('SELECT count(*) FROM quads').fetchone()[0]
    num_graph_elts = cur.execute('SELECT count(*) FROM graphelts').fetchone()[0]
    assert vx + eg + vp + ep == num_quads
    assert vx + eg + vp == num_graph_elts

    # Strings:
    st = cur.execute('SELECT count(*) FROM strings').fetchone()[0]

    cur.close()

    rc = RowCount()
    rc.qvl = vx
    rc.qe = eg
    rc.qvp = vp
    rc.qep = ep
    rc.s = st
    return rc


class RowCount:
    """Records counts of row types"""

    def __init__(self):
        # quads table, edge rows (p < 0)
        self.qe = 0
        # quads table, vertex label rows (p = 0)
        self.qvl = 0
        # quads table, vertex property rows (p > 0, s > 0)
        self.qvp = 0
        # quads table, edge property rows (p > 0, s < 0)
        self.qep = 0

        # graphelts table, vertex rows
        self.gv = 0
        # graphelts table, edge rows
        self.ge = 0
        # graphelts table, vertex property rows
        self.gvp = 0

        # strings table rows
        self.s = 0

    def __add__(self, other):
        rc = RowCount()

        rc.qe = self.qe + other.qe
        rc.qvl = self.qvl + other.qvl
        rc.qvp = self.qvp + other.qvp
        rc.qep = self.qep + other.qep

        rc.gv = self.gv + other.gv
        rc.ge = self.ge + other.ge
        rc.gvp = self.gvp + other.gvp

        rc.s = self.s + other.s

        return rc

    def __str__(self):
        parts = []
        for a in "qe qvl qvp qep gv ge gvp s".split():
            parts.append(f'{a}: {getattr(self, a)}')
        return f'({", ".join(parts)})'

    def matches(self, tup):
        """
        Pass a tuple of integers. We say whether our counts, in canonical order
        (i.e. order initialized in __init__() method), equal these, resp.

        :param tup: tuple of ints
        :return: boolean
        """
        return all(
            getattr(self, a) == tup[i]
            for i, a in enumerate("qe qvl qvp qep gv ge gvp s".split())
        )


def completely_remove_vertex(con: sqlite3.Connection, vertex_id):
    """
    Completely remove a vertex.

    This completely removes the vertex, as well as all incident edges, and all
    properties of the vertex and its incident edges. All necessary rows from both
    the quads table and the graphelts table are removed.

    :param con: connection
    :param vertex_id: the ID of the vertex to be removed
    :return: RowCount
    """
    rc = RowCount()

    # Vertex label:
    log_qqc('sp')
    cur = con.cursor()
    label_codes = cur.execute(
        "DELETE FROM quads WHERE s = ? AND p = 0 RETURNING o",
        [vertex_id]
    ).fetchall()
    cur.close()
    rc.qvl = len(label_codes)

    # Vertex ID:
    rc.gv = delete_elements_from_graphelts_table(con, vertex_id)

    # Vertex property quads:
    log_qqc('s', 'p')
    cur = con.cursor()
    rows = cur.execute(
        "DELETE FROM quads WHERE s = ? AND p > 0 RETURNING g",
        [vertex_id]
    ).fetchall()
    cur.close()
    vprop_ids = [r[0] for r in rows]
    rc.qvp = len(vprop_ids)

    # Vertex property IDs:
    rc.gvp = delete_elements_from_graphelts_table(con, vprop_ids)

    # Outgoing edges:
    log_qqc('s', 'p')
    cur = con.cursor()
    rows = cur.execute(
        "SELECT g FROM quads WHERE s = ? AND p < 0",
        [vertex_id]
    ).fetchall()
    cur.close()
    negative_edge_ids = [r[0] for r in rows]
    rc += completely_remove_edges(con, negative_edge_ids)

    # Incoming edges:
    log_qqc('o', 'p')
    cur = con.cursor()
    rows = cur.execute(
        "SELECT g FROM quads WHERE o = ? AND p < 0",
        [vertex_id]
    ).fetchall()
    cur.close()
    negative_edge_ids = [r[0] for r in rows]
    rc += completely_remove_edges(con, negative_edge_ids)

    return rc


def completely_remove_edges(con: sqlite3.Connection, negative_edge_ids):
    """
    Completely remove a set of edges.

    This completely removes the edges, as well as all of their properties.
    All necessary rows from both the quads table and the graphelts table are removed.

    :param con: connection
    :param negative_edge_ids: int or list of ints, being the *negative valence* IDs
        of the edges to be deleted. (I.e. you should be passing negative integers.)
    :return: RowCount
    """
    if isinstance(negative_edge_ids, int):
        negative_edge_ids = [negative_edge_ids]

    rc = RowCount()
    cur = con.cursor()

    # Edge quads:
    id_set = ','.join(str(n) for n in negative_edge_ids)
    log_qqc('g')
    rows = cur.execute(
        f"DELETE FROM quads WHERE g IN ({id_set}) RETURNING -g"
    ).fetchall()
    positive_ids_deleted = [r[0] for r in rows]
    rc.qe = len(positive_ids_deleted)

    # Edge property quads:
    log_qqc('s', 'p')
    eprops = cur.execute(
        f"DELETE FROM quads WHERE s IN ({id_set}) AND p > 0 RETURNING p"
    ).fetchall()
    rc.qep = len(eprops)

    cur.close()

    # Edge IDs:
    rc.ge = delete_elements_from_graphelts_table(con, positive_ids_deleted)

    return rc


def completely_remove_vertex_property(con: sqlite3.Connection, vprop_id):
    """
    Completely remove a vertex property.

    This removes a row from both the quads table and the graphelts table.

    :param con: connection
    :param vprop_id: the ID of the vertex property to be removed.
    :return: RowCount
    """
    rc = RowCount()

    cur = con.cursor()
    log_qqc('g')
    ids_deleted = cur.execute(
        "DELETE FROM quads WHERE g = ? RETURNING g", [vprop_id]
    ).fetchall()
    cur.close()
    rc.qvp = len(ids_deleted)

    rc.gvp = delete_elements_from_graphelts_table(con, vprop_id)

    return rc


def completely_remove_edge_property(con: sqlite3.Connection, edge_id, prop_name):
    """
    Completely remove an edge property.

    :param con: connection
    :param edge_id: the ID of the edge (pos or neg valence is okay) whose property
        is to be removed.
    :param prop_name: string, being the name of the property that is to be removed.
    :return: RowCount
    """
    rc = RowCount()

    if edge_id > 0:
        edge_id = -edge_id

    cur = con.cursor()
    log_qqc('sp')
    eprops = cur.execute(
        "DELETE FROM quads WHERE s = ? AND p IN (SELECT id FROM strings WHERE v = ?) RETURNING p",
        [edge_id, prop_name]
    ).fetchall()
    cur.close()
    rc.qep = len(eprops)

    return rc


def merge_string(con: sqlite3.Connection, val):
    """
    Retrieve the ID of an existing string, or add a new string and return its
    newly formed ID.

    :param con: connection
    :param val: string value
    :return: string ID
    """
    cur = con.cursor()
    r = cur.execute("SELECT id FROM strings WHERE v = ?", [val]).fetchone()
    if r is None:
        # The string is not yet present
        cur.execute("INSERT INTO strings (v) VALUES (?) RETURNING id", [val])
        sid = cur.fetchone()[0]
    else:
        # The string is already present.
        sid = r[0]
    cur.close()
    return sid


def string_to_id(con: sqlite3.Connection, val):
    """
    Retrieve the ID of a string.

    :param con: connection
    :param val: the string
    :return: int or None
    """
    cur = con.cursor()
    cur.execute("SELECT id FROM strings WHERE v = ?", [val])
    r = cur.fetchone()
    cur.close()
    return r if r is None else r[0]


def id_to_string(con: sqlite3.Connection, i):
    """
    Retrieve the string of a given ID.

    :param con: connection
    :param i: the ID
    :return: string or None
    """
    cur = con.cursor()
    cur.execute('SELECT v FROM strings WHERE id = ?', [i])
    r = cur.fetchone()
    cur.close()
    return r if r is None else r[0]


def get_string_id(con: sqlite3.Connection, val, merge=False):
    if merge:
        return merge_string(con, val)
    else:
        return string_to_id(con, val)


def encode_property_name(con: sqlite3.Connection, name, merge=False):
    """
    Turn a property name into an int that is ready to be recorded in the quads table.

    :param con: connection
    :param name: the property name
    :param merge: set True to add the name if not already present
    :return: int or None
    """
    return get_string_id(con, name, merge=merge)


def encode_property_value(con: sqlite3.Connection, value, merge=False):
    """
    Turn a property value into a constraint for the o column in the quads table.

    :param con: connection
    :param value: None, boolean, int in the range [-2^62, 2^62 - 1], float, string, or an instance of
        one of gremlinpython's `P` or `TextP` predicate classes.
    :param merge: only affects string values. Set True to add the string if not already present.
    :return: int or None
    """
    if value is False:
        return 0
    elif value is True:
        return 1
    elif value is None:
        return 2
    elif isinstance(value, int):
        if value < Constants.MIN_INT or value > Constants.MAX_INT:
            return None
        else:
            return value + Constants.MIN_INT
    elif isinstance(value, float):
        str_value = str(value)
        base_id = get_string_id(con, str_value, merge=merge)
        return None if base_id is None else base_id + Constants.MAX_INT
    elif isinstance(value, str):
        return get_string_id(con, value, merge=merge)
    elif isinstance(value, predicates.P):
        return get_predicate_column_constraint(con, value, ColumnRole.PROPERTY_VALUE)
    return None


def decode_property_value(con: sqlite3.Connection, o):
    """
    Given the integer representing a property value in the quads table o column,
    return the actual property value.

    :param con: connection
    :param o: int representing prop value
    :return: prop value
    """
    if o == 0:
        return False
    elif o == 1:
        return True
    elif o == 2:
        return None
    elif o < 0:
        return o - Constants.MIN_INT
    elif o > Constants.MAX_INT:
        base_id = o - Constants.MAX_INT
        str_value = id_to_string(con, base_id)
        return float(str_value)
    else:
        return id_to_string(con, o)


def encode_vertex_label(con: sqlite3.Connection, value, merge=False):
    if isinstance(value, str):
        return get_string_id(con, value, merge=merge)
    elif isinstance(value, predicates.P):
        return get_predicate_column_constraint(con, value, ColumnRole.VERTEX_LABEL)
    return None


def encode_edge_label(con: sqlite3.Connection, value, merge=False):
    if isinstance(value, str):
        i = get_string_id(con, value, merge=merge)
        return None if i is None else -i
    elif isinstance(value, predicates.P):
        return get_predicate_column_constraint(con, value, ColumnRole.EDGE_LABEL)
    return None


def encode_label(con: sqlite3.Connection, subject_type: SubjectTypes, value, merge=False):
    if subject_type == SubjectTypes.VERTICES:
        return encode_vertex_label(con, value, merge=merge)
    elif subject_type == SubjectTypes.EDGES:
        return encode_edge_label(con, value, merge=merge)


def decode_vertex_label(con: sqlite3.Connection, o_col_value):
    return id_to_string(con, o_col_value)


def decode_edge_label(con: sqlite3.Connection, p_col_value):
    return id_to_string(con, -p_col_value)


def get_vertex_label(con: sqlite3.Connection, vertex_id):
    log_qqc('sp')
    cur = con.cursor()
    label = cur.execute("SELECT v FROM strings WHERE id IN (SELECT o FROM quads WHERE s = ? AND p = ?)", [
        vertex_id, Constants.HAS_LABEL_PREDICATE
    ]).fetchone()[0]
    cur.close()
    return label


def get_edge_label(con: sqlite3.Connection, edge_id):
    log_qqc('g')
    cur = con.cursor()
    label = cur.execute("SELECT v FROM strings WHERE id IN (SELECT -p FROM quads WHERE g = ?)", [
        -edge_id
    ]).fetchone()[0]
    cur.close()
    return label


def get_label(con: sqlite3.Connection, obj_id: int, subject_type: SubjectTypes):
    if subject_type == SubjectTypes.VERTICES:
        return get_vertex_label(con, obj_id)
    elif subject_type == SubjectTypes.EDGES:
        return get_edge_label(con, obj_id)


class InterpolatedQuery(str):
    """
    Use this to represent query strings containing one or more question marks, and keep
    their list of parameters associated with them. Supports left and right addition with
    regular strings or other InterpolatedQuery instances, concatenating the parameter
    lists appropriately.
    """

    def __new__(cls, value, params=None):
        self = str.__new__(cls, value)
        self.value = value
        self.params = params or []
        return self

    def __str__(self):
        return f'({self.value}, {self.params})'  # pragma: no cover

    def __add__(self, other):
        if isinstance(other, InterpolatedQuery):
            return InterpolatedQuery(self.value + other.value, self.params + other.params)
        if isinstance(other, str):
            iq = InterpolatedQuery(other)
            return self + iq
        raise NotImplemented  # pragma: no cover

    def __radd__(self, other):
        if isinstance(other, str):
            iq = InterpolatedQuery(other)
            return iq + self
        raise NotImplemented  # pragma: no cover


def query_join(glue, queries):
    """
    Perform a string join in such a way that InterpolatedQuery's will work properly.

    :param glue: the string to connect the queries
    :param queries: list of queries (str or InterpolatedQuery instances)
    :return: str or InterpolatedQuery
    """
    if not queries:
        return ''
    result = queries[0]
    for q in queries[1:]:
        result += glue + q
    return result


def write_strings_to_ids_query(strings, id_valence=1):
    """
    Given a finite list of strings, write a query string selecting the IDs of those strings.

    :param strings: nonempty list of desired strings
    :param id_valence: set to -1 to select negated IDs
    :return: string
    """
    sel = '-id' if id_valence == -1 else 'id'
    qms = ','.join(['?']*len(strings))
    return InterpolatedQuery(f"SELECT {sel} FROM strings WHERE v IN ({qms})", strings)


class ColumnRole(Enum):
    EDGE = 'EDGE'
    VERTEX = 'VERTEX'

    EDGE_LABEL = 'EDGE_LABEL'
    VERTEX_LABEL = 'VERTEX_LABEL'

    PROPERTY_NAME = 'PROPERTY_NAME'
    PROPERTY_VALUE = 'PROPERTY_VALUE'


def is_text_predicate(obj):
    """
    Say whether a given object is a text predicate.

    Since `TextP.gt` etc. actually return instances of `P`, not `TextP`, we
    need a slightly more sophisticated check in order to determine if we are
    looking at a text predicate.

    :param obj: any object
    :return: boolean
    """
    return isinstance(obj, predicates.TextP) or (
            isinstance(obj, predicates.P) and
            isinstance(obj.value, str) and
            obj.operator in ['gt', 'lt', 'gte', 'lte']
    )


class PredicateHandler:

    def handle_predicate(self, pred: predicates.P):
        operator = pred.operator
        value = pred.value

        if is_text_predicate(pred):
            if operator == 'startingWith':
                return self.starting_with(value)
            elif operator == 'containing':
                return self.containing(value)
            elif operator == 'endingWith':
                return self.ending_with(value)
            elif operator == 'gt':
                return self.gt(value)
            elif operator == 'lt':
                return self.lt(value)
            elif operator == 'gte':
                return self.gte(value)
            elif operator == 'lte':
                return self.lte(value)

        elif isinstance(pred, predicates.P):
            if operator == 'within':
                return self.within(value)

    def starting_with(self, value):
        raise NotImplementedError  # pragma: no cover

    def containing(self, value):
        raise NotImplementedError  # pragma: no cover

    def ending_with(self, value):
        raise NotImplementedError  # pragma: no cover

    def gt(self, value):
        raise NotImplementedError  # pragma: no cover

    def lt(self, value):
        raise NotImplementedError  # pragma: no cover

    def gte(self, value):
        raise NotImplementedError  # pragma: no cover

    def lte(self, value):
        raise NotImplementedError  # pragma: no cover

    def within(self, value):
        raise NotImplementedError  # pragma: no cover


class PredicateColumnConstraintHandler(PredicateHandler):

    def __init__(self, con: sqlite3.Connection, column_role: ColumnRole):
        self.con = con
        self.column_role = column_role

        text_sign = '-' if column_role is ColumnRole.EDGE_LABEL else ''
        self.text_p_basic = f"SELECT {text_sign}id FROM strings WHERE v "

    def build_interp_text_query(self, ending, value):
        return InterpolatedQuery(self.text_p_basic + ending, [value])

    def starting_with(self, value):
        return self.build_interp_text_query("GLOB ?", f'{value}*')

    def containing(self, value):
        return self.build_interp_text_query("GLOB ?", f'*{value}*')

    def ending_with(self, value):
        return self.build_interp_text_query("GLOB ?", f'*{value}')

    def gt(self, value):
        return self.build_interp_text_query("> ?", value)

    def lt(self, value):
        return self.build_interp_text_query("< ?", value)

    def gte(self, value):
        return self.build_interp_text_query(">= ?", value)

    def lte(self, value):
        return self.build_interp_text_query("<= ?", value)

    def within(self, value):
        values = value
        con = self.con
        column_role = self.column_role

        if column_role == ColumnRole.EDGE:
            # We assume you passed a list of edge IDs, and at worst you
            # forgot to make them negative, but you did use consistent sign throughout.
            if len(values) > 0 and values[0] > 0:
                return [-v for v in values]
            else:
                return values
        elif column_role == ColumnRole.VERTEX:
            # We assume you passed a list of vertex IDs, and they're all positive.
            # No change required.
            return values

        # For the following cases, it's possible that `None` be returned when we attempt to
        # encode, due to strings not being present in the strings table. So we need to filter
        # those cases before returning.
        if column_role == ColumnRole.EDGE_LABEL:
            # We assume you passed a list of strings.
            values = [encode_edge_label(con, v) for v in values]
        elif column_role == ColumnRole.VERTEX_LABEL:
            # We assume you passed a list of strings.
            values = [encode_vertex_label(con, v) for v in values]
        elif column_role == ColumnRole.PROPERTY_NAME:
            # We assume you passed a list of strings.
            values = [encode_property_name(con, v) for v in values]
        elif column_role == ColumnRole.PROPERTY_VALUE:
            # We assume you passed a list of types that can be encoded as property values
            # (so None, bool, int, float, or str).
            values = [encode_property_value(con, v) for v in values]

        return [v for v in values if v is not None]


def get_predicate_column_constraint(con: sqlite3.Connection, pred: predicates.P, column_role: ColumnRole):
    """
    Generate a constraint that can be passed to the `get_quads()` function.

    :param con: connection
    :param pred: an instance of gremlinpython's `P` or `TextP` predicate classes
    :param column_role: a value of the ColumnRole enum class, indicating the
        role this predicate is playing
    :return: constraint that can be passed to the `get_quads()` function.
    """
    handler = PredicateColumnConstraintHandler(con, column_role)
    return handler.handle_predicate(pred)


def write_quads_query(sel="*", s=None, p=None, o=None, g=None):
    """
    Write an sqlite query against the quads table

    :param sel: what you want to SELECT from the quads table
    :param s: Constrain the s component of the quad:
        * None: no constraint
        * Sign.NEG or Sign.POS: require this sign
        * a single integer: require that it equal this value
        * a list of integers: require that it be in this set
        * string: require that it be in the set returned by this string as a subquery
    :param p: like s
    :param o: like s
    :param g: like s
    :return: query string
    """
    given = {'s': s, 'p': p, 'o': o, 'g': g}

    primary = []
    secondary = []

    conditions = {}
    for k, v in given.items():
        if v is not None:
            if v is Sign.NEG:
                conditions[k] = f'{k} < 0'
                secondary.append(k)
            elif v is Sign.POS:
                conditions[k] = f'{k} > 0'
                secondary.append(k)
            elif isinstance(v, int):
                conditions[k] = f'{k} = {v}'
                primary.append(k)
            elif isinstance(v, list):
                conditions[k] = f'{k} IN ({",".join(str(n) for n in v)})'
                primary.append(k)
            elif isinstance(v, str):
                # Use explicit `+` here in case v is an InterpolatedQuery:
                conditions[k] = f'{k} IN (' + v + ')'
                primary.append(k)

    log_qqc(primary, secondary)

    condition_seq = list(conditions.values())

    where_clause = ''
    if condition_seq:
        where_clause = ' WHERE ' + query_join(' AND ', condition_seq)

    query = f"SELECT {sel} FROM quads" + where_clause

    return query


def run_quads_query(con: sqlite3.Connection, sel="*", s=None, p=None, o=None, g=None, do_close=False, fetch_one=False):
    query = write_quads_query(sel, s, p, o, g)
    return run_query(con, query, do_close=do_close, fetch_one=fetch_one)


def run_query(con: sqlite3.Connection, query, do_close=False, fetch_one=False):
    cur = con.cursor()
    if isinstance(query, InterpolatedQuery):
        cur.execute(query.value, query.params)
    else:
        cur.execute(query)
    if do_close:
        rows = [cur.fetchone()] if fetch_one else cur.fetchall()
        cur.close()
        return DummyCursor(rows)
    else:
        return cur


# We build a dictionary mapping all known expected quad constraint patterns to the
# index we expect to be used for each.
_ix_pairs = """
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
""".split()
expected_quad_constraint_patterns = dict(zip(_ix_pairs[0::2], _ix_pairs[1::2]))


qqc_header = 'QUAD QUERY CONSTRAINTS: '


def format_qqc(primary, secondary=''):
    """
    Write a string of the form "[...|...]" to describe quad query constraints.

    The letters in each half will be sorted for you, so you can pass them in any order.

    :param primary: string or list of column names receiving exact constraints
    :param secondary: optional string or list of column names receiving inexact constraints
    :return: string
    """
    return f'[{"".join(sorted(primary, reverse=True))}|{"".join(sorted(secondary, reverse=True))}]'


def log_qqc(primary, secondary=''):
    """
    Log a message noting quad query constraints.

    :param primary: string or list of column names
    :param secondary: optional string or list of column names
    :return: nothing
    """
    qqc = format_qqc(primary, secondary)
    msg = qqc_header + qqc
    logger.info(msg)


class QqcPatternFilter(logging.Filter):
    """
    Raise exception on presence of unexpected QQC pattern.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        t = record.msg
        if t.startswith(qqc_header):
            i0 = len(qqc_header)
            i1 = t.index(']')
            pattern = t[i0:i1+1]
            if pattern not in expected_quad_constraint_patterns:
                raise UnexpectedQuadConstraintPattern(pattern)
        return False


qqc_handler = logging.StreamHandler()
qqc_handler.addFilter(QqcPatternFilter())


def check_qqc_patterns():
    """
    Invoke if you want exceptions to be raised on unexpected QQC patterns.
    """
    logger.addHandler(qqc_handler)
