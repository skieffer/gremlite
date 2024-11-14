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

from collections import deque
from contextlib import contextmanager
from functools import cmp_to_key
import random
import sqlite3
from typing import Optional, List

from gremlin_python.process.traversal import Bytecode, Cardinality, Order, P

from .cursors import DummyCursor, GeneratorCursor
from .errors import BadStepArgs, BadStepCombination, UnknownStep
from .filtering import (
    FilterHandler, FilterableElement, FilterQueryBuilder, FilterableQuery, heuristic_filter_split,
)
from .base import (
    GremliteConfig, CanonicalStepNames, ProducerCursor,
    Producer, SingletonProducer, RestartBarrier, PassthroughBarrier, EvaluatorProducer,
    ModulatingProducer, ModulatingEvaluatorProducer, ModulatingElementEvaluatorProducer,
    SideEffect, FilterProducer,
    ElementEvaluatorProducer, ElementFilterProducer,
    VertexConsumerProducer, EdgeConsumerProducer,
    PropertyConsumerProducer, ListConsumerProducer,
    StepStream, Step, TraversalContext
)
import gremlite.querytools as qt
from .querytools import PredicateHandler
from .results import Result


class FilteringProducer(Producer):
    """
    A producer that performs an SQLite query, and may combine *some* subsequent `has()` steps
    into this query, while leaving others to be performed on their own.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.query_builder = None
        self.initial_query_is_broad = False
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.special_build(ss)

        self.c_steps = ss.take_contributing_steps(filter_steps=True)
        filter_steps = self.c_steps.filter_steps

        if self.config.use_basic_heuristic_filtering:
            query_steps, element_steps = self.heuristic_filter_split(filter_steps)
        else:
            query_steps, element_steps = filter_steps, []

        self.query_builder = FilterQueryBuilder(self.config, self.con, query_steps)
        ss.insert_steps(element_steps)

    def make_next_cursor(self):
        fq = self.write_initial_subquery()
        query = self.query_builder.build(fq)
        return qt.run_query(self.con, query, do_close=self.config.read_all_at_once)

    def heuristic_filter_split(self, incoming_steps: List[Step]) -> (List[Step], List[Step]):
        return heuristic_filter_split(incoming_steps, initial_query_is_broad=self.initial_query_is_broad)

    def special_build(self, ss: StepStream):
        """Subclasses may implement, to perform special steps at the start of `build()`. """
        pass  # pragma: no cover

    def write_initial_subquery(self) -> FilterableQuery:
        raise NotImplementedError  # pragma: no cover

    def assemble_outgoing_result(self, next_cursor_item) -> Result:
        """
        :param next_cursor_item: row of length one, holding the signed ID for a graph element,
            i.e. a positive integer for a vertex, negative for an edge.
        :return: Result
        """
        raise NotImplementedError  # pragma: no cover


class V(FilteringProducer):
    """
    V() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.vids = None
        self.instances = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.vids, self.instances = self.base_step.extract_list_of_ids(accept_vertices=True)
        self.initial_query_is_broad = not self.vids

    def write_initial_subquery(self) -> FilterableQuery:
        if self.query_builder.filter_steps:
            query = self.vids or qt.Sign.POS
        else:
            query = qt.write_quads_query(
                sel='s', s=self.vids or None, p=qt.Constants.HAS_LABEL_PREDICATE
            )
        return FilterableQuery(qt.SubjectTypes.VERTICES, query)

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()

        if self.instances:
            r.note_vertices(self.instances)
            self.instances = None

        vid = next_cursor_item[0]
        r.add_vertex_to_path(vid)
        return r


class E(FilteringProducer):
    """
    E() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.eids = None
        self.instances = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.eids, self.instances = self.base_step.extract_list_of_ids(accept_edges=True)
        self.initial_query_is_broad = not self.eids

    def write_initial_subquery(self) -> FilterableQuery:
        negative_eids = [-eid for eid in self.eids]
        if self.query_builder.filter_steps:
            query = negative_eids or qt.Sign.NEG
        else:
            query = qt.write_quads_query(
                sel='g', g=negative_eids or qt.Sign.NEG
            )
        return FilterableQuery(qt.SubjectTypes.EDGES, query)

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()

        if self.instances:
            r.note_edges(self.instances)
            self.instances = None

        g = next_cursor_item[0]
        eid = -g
        r.add_edge_to_path(eid)
        return r


class AddV(Producer):
    """
    add_v() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.label = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_string_varargs(allowed_numbers=[0, 1], noun='label')
        self.label = args[0] if args else 'vertex'

    def make_next_cursor(self):
        vertex_id = qt.create_vertex(self.con, self.label)
        return DummyCursor([[vertex_id]])

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()
        vertex_id = next_cursor_item[0]
        r.add_vertex_to_path(vertex_id, label=self.label)
        return r


class AddE(VertexConsumerProducer):
    """
    add_e() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.edge_label = None
        self.source_obj_label = None
        self.target_obj_label = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_string_varargs(allowed_numbers=[1], noun='label')
        self.edge_label = args[0]

        self.c_steps = ss.take_contributing_steps(new_edge_steps=True)
        self.source_obj_label = self.c_steps.get_source()
        self.target_obj_label = self.c_steps.get_target()

    def check_incoming_result_type(self, err_msg_extension=''):
        if self.source_obj_label is None or self.target_obj_label is None:
            super().check_incoming_result_type(err_msg_extension=', since not both source and target were given')

        for obj_label in [self.source_obj_label, self.target_obj_label]:
            if obj_label is not None:
                if not self.current_incoming_result.has_labeled_vertex(obj_label):
                    raise BadStepCombination(self.build_err_msg(
                        f'named "{obj_label}" as endpoint, but object cannot be found.'
                    ))

    @property
    def source_vertex(self):
        return (
            self.in_r.last_object if self.source_obj_label is None
            else self.in_r.get_labeled_object(self.source_obj_label)
        )

    @property
    def target_vertex(self):
        return (
            self.in_r.last_object if self.target_obj_label is None
            else self.in_r.get_labeled_object(self.target_obj_label)
        )

    def make_next_cursor(self):
        edge_id = qt.create_edge(
            self.con, self.source_vertex.id, self.edge_label, self.target_vertex.id
        )
        return DummyCursor([[edge_id]])

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()
        edge_id = next_cursor_item[0]
        r.add_edge_to_path(edge_id, self.source_vertex.id, self.target_vertex.id, label=self.edge_label)
        return r


class As(SideEffect):
    """
    as() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.obj_labels = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.obj_labels = self.base_step.extract_string_varargs(nonzero=True, noun='labels')

    def side_effect(self):
        if self.context is TraversalContext.BASIC:
            for label in self.obj_labels:
                self.in_r.label_last_object(label)

    @property
    def do_forward(self):
        if self.context is TraversalContext.PATTERN_MATCHING:
            obj = self.in_r.last_object
            return all(
                obj is self.in_r.get_labeled_object(label)
                for label in self.obj_labels
            )
        else:
            return True


class Count(RestartBarrier):
    """
    count() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.c = 0
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def for_each_incoming_result(self, r: Result):
        self.c += 1

    @property
    def final_value(self):
        return self.c


class Fold(RestartBarrier):
    """
    fold() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.L = []
        super().__init__(con, ss, parent=parent)
    
    def build(self, ss: StepStream):
        self.base_step.check_no_args()
        
    def for_each_incoming_result(self, r: Result):
        self.L.append(r.last_object)

    @property
    def final_value(self):
        return self.L


class Barrier(PassthroughBarrier):
    """
    barrier() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()


class Store(ModulatingProducer):
    """
    store() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.list_name = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        args = self.base_step.extract_string_varargs(allowed_numbers=[1])
        self.list_name = args[0]

    def make_next_cursor(self):
        obj = self.in_r.last_object
        m = self.next_modulator()
        if m is not None:
            with carry_on(self, m) as cur:
                if cur.has_next():
                    obj = cur.fetchone().last_object
                else:
                    obj = None
        r = self.incoming_result_or_new()
        if obj is not None:
            r.add_to_storage_list(self.list_name, obj)
        return DummyCursor([r])


class Cap(RestartBarrier):
    """
    cap() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.list_name = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_string_varargs(allowed_numbers=[1])
        self.list_name = args[0]

    @property
    def final_value(self):
        r = self.incoming_result_or_new()
        return r.get_storage_list(self.list_name)


class OrderStep(ModulatingProducer, PassthroughBarrier):
    """
    order() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.orderings = []
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.base_step.check_no_args()

    def preprocess_by_step(self, step: Step) -> Step:
        ordering = Order.asc
        args = step.args
        if args and isinstance(args[-1], Order):
            ordering = args[-1]
            # We work with a copy of the Step, instead of the original, so that
            # any error messages generated from the Step reflect what the user
            # passed in.
            step = step.copy()
            step.args = args[:-1]
        self.orderings.append(ordering)
        return step

    def compare(self, A: Result, B: Result):
        for a, b, ordering in zip(A.order_keys, B.order_keys, self.orderings):
            if a < b:
                return -1 if ordering is Order.asc else 1
            elif b < a:
                return 1 if ordering is Order.asc else -1
        return 0

    def make_results_list(self):
        # In case there were no `by()` steps, we still need an ordering:
        if len(self.orderings) == 0:
            self.orderings.append(Order.asc)

        # Non-productive sorting modulators are supposed to filter,
        # even if they turn out not to be needed as secondary keys,
        # and even if we are shuffling.
        # Therefore, we start by applying all of them, to all incoming results.
        filtered_results = []
        for r0 in self.all_incoming_results:
            assert isinstance(r0, Result)
            r0.clear_order_keys()
            for modulator in self.modulators:
                if modulator is None:
                    key = r0.last_object_sort_key
                else:
                    with carry_on(self, modulator, from_result=r0) as cur:
                        if cur.has_next():
                            r1 = cur.fetchone()
                            key = r1.last_object_sort_key
                        else:
                            break
                r0.add_order_key(key)
            else:
                # We never broke, i.e. every modulator was productive.
                filtered_results.append(r0)

        if Order.shuffle in self.orderings:
            # We implement the rule that, "If anything's a shuffle, it's all a shuffle."
            random.shuffle(filtered_results)
        else:
            # Otherwise, there is no shuffle. Everything is either ascending or descending.
            filtered_results.sort(
                key=cmp_to_key(lambda A, B: self.compare(A, B))
            )

        return filtered_results


class Unfold(ListConsumerProducer):
    """
    unfold() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def make_next_cursor(self):
        return DummyCursor(self.in_r.last_object)

    def assemble_outgoing_result(self, next_cursor_item) -> Result:
        self.in_r.move_to_value(next_cursor_item)
        return self.in_r


class Key(EvaluatorProducer, PropertyConsumerProducer):
    """
    key() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def evaluate(self, r: Result):
        return r.last_object.key


class Value(EvaluatorProducer, PropertyConsumerProducer):
    """
    key() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def evaluate(self, r: Result):
        return r.last_object.value


class ValueMap(ModulatingProducer, ElementEvaluatorProducer):
    """
    value_map() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.prop_names = None
        self.incl_tokens = False
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        base_step = self.base_step

        args = base_step.args
        if args and isinstance(args[0], bool):
            self.incl_tokens = args[0]
            # We work with a copy of the Step, instead of the original, so that
            # any error messages generated from the Step reflect what the user
            # passed in.
            base_step = base_step.copy()
            base_step.args = args[1:]

        args = base_step.extract_string_varargs(noun='property names')
        self.prop_names = args or None

        ModulatingProducer.build(self, ss)

    def evaluate(self, r: Result):
        obj = r.last_object

        d0 = r.build_element_info_dict(
            obj, incl_tokens=self.incl_tokens, prop_names=self.prop_names
        )

        d1 = {}
        for k, v in d0.items():
            modulator = self.next_modulator()
            if modulator is not None:
                r0 = Result(self.con, self.ss)
                r0.move_to_value(v)
                with carry_on(self, modulator, from_result=r0) as cur:
                    if cur.has_next():
                        r1 = cur.fetchone()
                        v = r1.last_object
                    else:
                        continue
            d1[k] = v

        return d1


class ElementMap(ElementEvaluatorProducer):
    """
    element_map() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.prop_names = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_string_varargs(noun='property names')
        self.prop_names = args or None

    def evaluate(self, r: Result):
        obj = r.last_object
        return r.build_element_info_dict(
            obj, incl_tokens=True, incl_endpts=True,
            prop_names=self.prop_names, single_props=True
        )


class Constant(ElementEvaluatorProducer):
    """
    constant() step
    """

    def build(self, ss: StepStream):
        if len(self.base_step.args) != 1:
            raise BadStepArgs(self.build_err_msg("Pass exactly one arg."))
        a = self.base_step.args[0]
        # Any constraints on the type of the constant?
        # For now, we'll just say it can't be Bytecode.
        if isinstance(a, Bytecode):
            raise NotImplementedError

    def evaluate(self, r: Result):
        return self.base_step.args[0]


class Id(ElementEvaluatorProducer):
    """
    id() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        super().__init__(con, ss, parent=parent)
        self.accept_vertex_properties = True

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def evaluate(self, r: Result):
        return r.last_object.id


class Label(ElementEvaluatorProducer):
    """
    label() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def evaluate(self, r: Result):
        obj = r.last_object
        r.determine_label(obj)
        return obj.label


class NoneStep(Producer):
    """
    none() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def make_next_cursor(self):
        return DummyCursor.emtpy_cursor()


class Identity(FilterProducer):
    """
    identity() step
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def passes(self, r: Result):
        return True


class SimplePath(FilterProducer):
    """
    simple_path()
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def passes(self, r: Result):
        return r.path_is_simple()


class Not(FilterProducer):
    """
    not() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecode = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_bytecode_varargs(allowed_numbers=[1])
        self.bytecode = args[0]

    def passes(self, r: Result):
        with carry_on(self, self.bytecode) as cur:
            return not cur.has_next()


class And(FilterProducer):
    """
    and() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecodes = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.bytecodes = self.base_step.extract_bytecode_varargs(nonzero=True)

    def passes(self, r: Result):
        for bc in self.bytecodes:
            with carry_on(self, bc) as cur:
                if not cur.has_next():
                    return False
        return True


class Where(FilterProducer):
    """
    where() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecode = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.bytecode = self.base_step.extract_bytecode_varargs(allowed_numbers=[1])[0]

    def passes(self, r: Result):
        with carry_on(self, self.bytecode, context=TraversalContext.PATTERN_MATCHING) as cur:
            return cur.has_next()


class Or(FilterProducer):
    """
    or() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecodes = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.bytecodes = self.base_step.extract_bytecode_varargs(nonzero=True)

    def passes(self, r: Result):
        for bc in self.bytecodes:
            with carry_on(self, bc) as cur:
                if cur.has_next():
                    return True
        return False


class Limit(FilterProducer):
    """
    limit() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.counter = 0
        self.limit = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_varargs_of_types([int], allowed_numbers=[1])
        self.limit = args[0]

    def passes(self, r: Result):
        self.counter += 1
        # `limit()` is not a pure filter step; when the limit is reached, it has
        # to forcefully stop everything by raising `StopIteration`. (If it just
        # returned an empty cursor, we'd continue trying again with a new incoming
        # result.)
        if self.counter <= self.limit:
            return True
        else:
            raise StopIteration


class Has(ElementFilterProducer, FilterHandler, PredicateHandler):
    """
    has() step

    This handles the case in which a `has()` step operates standalone.
    Other producers may consume some `has()` steps as contributing steps, while
    leaving others to be handled by this class.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.given_strings = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        pass

    def passes(self, r: Result):
        elt0 = FilterableElement(r)
        elt1 = self.dispatch(self.base_step, elt0)
        return elt1 is elt0

    def has_label(self, filter_step: Step, labels, filterable: FilterableElement) -> FilterableElement:
        r = filterable.result
        obj = r.last_object
        r.determine_label(obj)

        if labels and isinstance(labels[0], P):
            pred = labels[0]
            self.given_strings = [obj.label]
            passes = self.handle_predicate(pred)
        else:
            passes = obj.label in labels

        return filterable if passes else FilterableElement(None)

    def unary_has(self, filter_step: Step, key: str, filterable: FilterableElement) -> FilterableElement:
        r = filterable.result
        obj = r.last_object
        d = r.get_properties(obj, prop_names=[key], single=True, value_only=True)
        return filterable if d else FilterableElement(None)

    def binary_has(self, filter_step: Step, key: str, value, filterable: FilterableElement) -> FilterableElement:
        r = filterable.result
        obj = r.last_object
        values = r.get_properties(obj, prop_names=[key], value_only=True, as_list=True)

        if isinstance(value, P):
            pred = value
            self.given_strings = values
            passes = self.handle_predicate(pred)
        else:
            passes = value in values

        return filterable if passes else FilterableElement(None)

    def starting_with(self, value):
        return any(s.startswith(value) for s in self.given_strings)

    def containing(self, value):
        return any(s.find(value) >= 0 for s in self.given_strings)

    def ending_with(self, value):
        return any(s.endswith(value) for s in self.given_strings)

    def gt(self, value):
        return any(s > value for s in self.given_strings)

    def lt(self, value):
        return any(s < value for s in self.given_strings)

    def gte(self, value):
        return any(s >= value for s in self.given_strings)

    def lte(self, value):
        return any(s <= value for s in self.given_strings)

    def within(self, value):
        return any(s in value for s in self.given_strings)


"""
has_label() step
"""
HasLabel = Has


class IncidentEdgeTraverser(VertexConsumerProducer, FilteringProducer):
    """
    Abstract base class for steps that move from a vertex either to
    an incident edge, or across an incident edge to an adjacent vertex.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.specific_labels = None
        self.p = None
        self.starting_vertex_id = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.specific_labels = self.base_step.extract_string_varargs(noun='edge labels')
        self.p = (
            qt.write_strings_to_ids_query(self.specific_labels, id_valence=-1)
            if self.specific_labels else qt.Sign.NEG
        )

    @property
    def sel(self):
        raise NotImplementedError  # pragma: no cover

    @property
    def s(self):
        raise NotImplementedError  # pragma: no cover

    @property
    def o(self):
        raise NotImplementedError  # pragma: no cover

    @property
    def subject_type(self):
        return qt.SubjectTypes.EDGES if 'g' in self.sel else qt.SubjectTypes.VERTICES

    def write_initial_subquery(self) -> FilterableQuery:
        self.starting_vertex_id = self.in_r.last_object.id

        base_queries = []
        for sel, s, o in zip(self.sel, self.s, self.o):
            base_queries.append(qt.write_quads_query(sel=sel, s=s, p=self.p, o=o))
        query = qt.query_join(' UNION ALL ', base_queries)

        return FilterableQuery(self.subject_type, query)

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()
        signed_id = next_cursor_item[0]
        r.add_object_to_path(signed_id)
        return r


class BothE(IncidentEdgeTraverser):
    """
    both_e() step. Move from the current vertex to both its incoming and its outgoing edges.
    """

    @property
    def sel(self):
        return ['g', 'g']

    @property
    def s(self):
        return [None, self.starting_vertex_id]

    @property
    def o(self):
        return [self.starting_vertex_id, None]


class InE(IncidentEdgeTraverser):
    """
    in_e() step. Move from the current vertex to its incoming edges.
    """

    @property
    def sel(self):
        return ['g']

    @property
    def s(self):
        return [None]

    @property
    def o(self):
        return [self.starting_vertex_id]


class OutE(IncidentEdgeTraverser):
    """
    out_e() step. Move from the current vertex to its outgoing edges.
    """

    @property
    def sel(self):
        return ['g']

    @property
    def s(self):
        return [self.starting_vertex_id]

    @property
    def o(self):
        return [None]


class Both(IncidentEdgeTraverser):
    """
    both() step. Hop from the current vertex to neighboring vertices, along both incoming
    and outgoing edges.
    """

    @property
    def sel(self):
        return ['s', 'o']

    @property
    def s(self):
        return [None, self.starting_vertex_id]

    @property
    def o(self):
        return [self.starting_vertex_id, None]


class In(IncidentEdgeTraverser):
    """
    in() step. Hop from the current vertex to neighboring vertices, along incoming edges.
    """

    @property
    def sel(self):
        return ['s']

    @property
    def s(self):
        return [None]

    @property
    def o(self):
        return [self.starting_vertex_id]


class Out(IncidentEdgeTraverser):
    """
    out() step. Hop from the current vertex to neighboring vertices, along outgoing edges.
    """

    @property
    def sel(self):
        return ['o']

    @property
    def s(self):
        return [self.starting_vertex_id]

    @property
    def o(self):
        return [None]


class EdgeToEndpointTraverser(EdgeConsumerProducer):
    """
    Abstract base class for the steps out_v(), in_v(), other_v(), both_v(), that move
    from an edge to one or both of its endpoint vertices.
    """

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    def make_next_cursor(self):
        ids = [v.id for v in self.goal_vertices]
        return DummyCursor(ids)

    @property
    def goal_vertices(self):
        """
        Subclasses must return the list of vertices we want to move to.
        """
        raise NotImplementedError  # pragma: no cover

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()
        # Because of the way Edge objects are formed in the Result object,
        # the Edge will always already have references to its two endpoint vertices.
        vertex_id = next_cursor_item
        r.add_vertex_to_path(vertex_id)
        return r


class InV(EdgeToEndpointTraverser):
    """
    in_v() step (move from an edge to its target vertex)
    """

    @property
    def goal_vertices(self):
        return [self.in_r.last_object.inV]


class OutV(EdgeToEndpointTraverser):
    """
    out_v() step (move from an edge to its source vertex)
    """

    @property
    def goal_vertices(self):
        return [self.in_r.last_object.outV]


class BothV(EdgeToEndpointTraverser):
    """
    both_v() step (move from an edge to both of its adjacent
    vertices)
    """

    @property
    def goal_vertices(self):
        edge = self.in_r.last_object
        return [edge.outV, edge.inV]


class OtherV(EdgeToEndpointTraverser):
    """
    other_v() step (move from an edge to that one of its endpoints
    that we did *not* just visit
    """

    @property
    def goal_vertices(self):
        edge = self.in_r.last_object
        origin_vertex = self.in_r.penultimate_object
        both_vertices = [edge.outV, edge.inV]
        if origin_vertex not in both_vertices:
            msg = "Previous object not one of edge's endpoints."
            raise BadStepCombination(self.build_err_msg(msg))
        both_vertices.remove(origin_vertex)
        other_vertex = both_vertices.pop()
        return [other_vertex]


class Property(SideEffect):
    """
    property() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.card = Cardinality.single
        self.prop_name = None
        self.prop_value = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.args

        if len(args) == 3:
            c = args[0]
            if not isinstance(c, Cardinality):
                raise BadStepArgs(self.build_err_msg("When passing 3 args, first should be a Cardinality"))
            self.card = c
            args = args[1:]

        if not len(args) == 2:
            raise BadStepArgs(self.build_err_msg("Pass either (Cardinality, key, value) or (key, value)."))

        prop_name, prop_value = args

        if not isinstance(prop_name, str):
            raise BadStepArgs(self.build_err_msg("property name should be string."))

        if not isinstance(prop_value, (int, float, bool, str, type(None))):
            raise BadStepArgs(self.build_err_msg("property value should be int, float, bool, str, or None"))

        self.prop_name = prop_name
        self.prop_value = prop_value

    def side_effect(self):
        r = self.in_r
        r.set_property(r.last_object, self.prop_name, self.prop_value, cardinality=self.card)


class Properties(ElementEvaluatorProducer):
    """
    properties() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.specific_names = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.specific_names = self.base_step.extract_string_varargs(noun='property names') or None

    @property
    def is_multi(self):
        return True

    def evaluate(self, r: Result):
        return r.get_properties(r.last_object, prop_names=self.specific_names, as_list=True)


class Values(Properties):
    """
    values() step
    """

    def evaluate(self, r: Result):
        props = super().evaluate(r)
        return [p.value for p in props]


class Coalesce(Producer):
    """
    coalesce() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecodes = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.bytecodes = self.base_step.extract_bytecode_varargs()

    def make_next_cursor(self):
        return GeneratorCursor(self.follow_first_successful_traversal())

    def follow_first_successful_traversal(self):
        """
        Follow the first traversal (if any) that produces one or more results.

        :return: generator
        """
        for bc in self.bytecodes:
            with carry_on(self, bc) as cur:
                if cur.has_next():
                    while cur.has_next():
                        yield cur.fetchone()
                    break


class Union(Producer):
    """
    union() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecodes = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.bytecodes = self.base_step.extract_bytecode_varargs()

    def make_next_cursor(self):
        return GeneratorCursor(self.follow_all_traversals())

    def follow_all_traversals(self):
        """
        Follow each traversal on the incoming result, and yield results.

        :return: generator
        """
        for bc in self.bytecodes:
            with carry_on(self, bc) as cur:
                while cur.has_next():
                    yield cur.fetchone()


class FlatMap(Union):
    """
    flat_map() step
    """

    def build(self, ss: StepStream):
        self.bytecodes = self.base_step.extract_bytecode_varargs(allowed_numbers=[1])


class Select(ModulatingEvaluatorProducer):
    """
    select() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.obj_labels = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.obj_labels = self.base_step.extract_string_varargs(nonzero=True, noun='step labels')

    def evaluate(self, r: Result):
        r0 = r
        d = {}

        for obj_label in self.obj_labels:
            obj = r0.get_labeled_object(obj_label)

            # In TinkerGraph, if you try to select a label that doesn't exist,
            # it's not an error; you just get no result.
            if obj is None:
                raise StopIteration

            modulator = self.next_modulator()

            if modulator is not None:
                assert isinstance(modulator, Bytecode)
                # Follow the traversal, and accept the first result.
                # If there are no results, then the `select()` step returns no results.
                # Question: could we make `mixed_copy()` work here instead of full `copy()`?
                r = r0.copy(stop_at_latest=obj_label)
                with carry_on(self, modulator, from_result=r) as cur:
                    if cur.has_next():
                        result = cur.fetchone()
                        obj = result.last_object
                    else:
                        raise StopIteration

            d[obj_label] = obj

        # If only one thing was selected, return it as a bare value.
        # Otherwise, we return a whole dictionary.
        if len(d) == 1:
            _, d = d.popitem()

        return d


class Path(ModulatingProducer, EvaluatorProducer):
    """
    path() step
    """

    def special_build(self, ss: StepStream):
        self.base_step.check_no_args()

    def evaluate(self, r: Result):
        r0 = r
        N = len(r0)

        objects = []
        for k in range(N):
            modulator = self.next_modulator()
            if modulator is None:
                obj = r0.get_object_by_index(k)
            else:
                # Question: could we make `mixed_copy()` work here instead of full `copy()`?
                r = r0.copy(rewind=N-1-k)
                with carry_on(self, modulator, from_result=r) as cur:
                    if cur.has_next():
                        obj = cur.fetchone().last_object
                    else:
                        # In Tinkergraph, if any modulator is non-productive,
                        # then we get no path at all.
                        raise StopIteration
            objects.append(obj)

        return r0.make_path(objects)


class Project(ModulatingElementEvaluatorProducer):
    """
    project() step
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.dict_keys = None
        super().__init__(con, ss, parent=parent)

    def special_build(self, ss: StepStream):
        self.dict_keys = self.base_step.extract_string_varargs(nonzero=True, noun='dictionary keys')

    def evaluate(self, r: Result):
        d = {}

        for k in self.dict_keys:
            modulator = self.next_modulator()
            if modulator is None:
                obj = self.in_r.last_object
            else:
                with carry_on(self, modulator) as cur:
                    if cur.has_next():
                        obj = cur.fetchone().last_object
                    else:
                        obj = None
            if obj is not None:
                d[k] = obj

        return d


class RepeatQueueItem:
    """
    Each item in the BFS queue for a `repeat()` step records the Result that was reached, and the
    number of applications of our main traversal that it took to get there.
    """

    def __init__(self, result: Result, num_apps: int):
        self.result = result
        self.num_apps = num_apps


class RepeatEmitUntil(Producer):
    """
    repeat() step, with optional emit(), until(), and times() modifiers.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.do_emit = False
        self.main_trav = None
        self.rep_limit = None
        self.until_trav = None

        self.emit_incoming = False
        self.until_incoming = False

        self.bfs_queue = deque()

        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        ss.rewind()
        self.c_steps = ss.take_contributing_steps(allow_repeats=False, repeater_steps=True)
        all_steps = self.c_steps.repeater_steps

        post_repeat = False

        for step in all_steps:
            if step.name == CanonicalStepNames.emit:
                step.check_no_args()
                self.do_emit = True
                if not post_repeat:
                    self.emit_incoming = True
            elif step.name == CanonicalStepNames.repeat:
                args = step.extract_bytecode_varargs(allowed_numbers=[1])
                self.main_trav = args[0]
                post_repeat = True
            elif step.name == CanonicalStepNames.times:
                args = step.extract_varargs_of_types([int], allowed_numbers=[1])
                self.rep_limit = args[0]
            elif step.name == CanonicalStepNames.until:
                args = step.extract_bytecode_varargs(allowed_numbers=[1])
                self.until_trav = args[0]
                if not post_repeat:
                    self.until_incoming = True

        if not post_repeat:
            raise BadStepCombination(self.build_err_msg("Missing `repeat()` step."))

    def enqueue(self, result: Result, num_apps: int):
        self.bfs_queue.append(RepeatQueueItem(result, num_apps))

    def make_next_cursor(self):
        self.bfs_queue.clear()
        self.enqueue(self.in_r.mixed_copy(), 0)
        cur = GeneratorCursor(self.bfs())
        return cur

    def bfs(self):
        """
        This method is designed to return a generator when invoked. So it uses `yield`
        to return results, and may return `None` to indicate that there are no more
        results to be generated.

        :return: generator
        """
        while self.bfs_queue:
            q = self.bfs_queue.popleft()
            r0 = q.result
            apps = q.num_apps

            do_emit = (apps == 0 and self.emit_incoming) or (apps > 0 and self.do_emit)
            do_check_until = (apps == 0 and self.until_incoming) or (apps > 0 and self.until_trav is not None)
            met_until_check = False

            if do_check_until:
                with carry_on(self, self.until_trav, from_result=r0) as cur:
                    if cur.has_next():
                        met_until_check = True

            if do_emit or met_until_check:
                yield r0

            if not met_until_check and (self.rep_limit is None or apps < self.rep_limit):
                with carry_on(self, self.main_trav, from_result=r0) as cur:
                    n = apps + 1
                    for r in cur.producer:
                        r1 = r.mixed_copy()
                        self.enqueue(r1, n)


class Drop(SideEffect):

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        super().__init__(con, ss, parent=parent)
        self.accept_vertex_properties = True
        self.accept_edge_properties = True

    def build(self, ss: StepStream):
        self.base_step.check_no_args()

    @property
    def do_forward(self):
        return False

    def side_effect(self):
        r = self.in_r
        if self.incoming_result_id_valence == 1:
            qt.completely_remove_vertex(self.con, r.last_object.id)
        elif self.incoming_result_id_valence == -1:
            qt.completely_remove_edges(self.con, -r.last_object.id)
        elif self.incoming_subject_type == qt.SubjectTypes.VERTICES:
            qt.completely_remove_vertex_property(self.con, r.last_object.id)
        elif self.incoming_subject_type == qt.SubjectTypes.EDGES:
            prop = r.last_object
            qt.completely_remove_edge_property(self.con, prop.element.id, prop.key)


class SideEffectStep(SideEffect):

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.bytecode = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        args = self.base_step.extract_bytecode_varargs(allowed_numbers=[1])
        self.bytecode = args[0]

    def side_effect(self):
        with carry_on(self, self.bytecode) as cur:
            if cur.has_next():
                cur.fetchone()


################################################################################

# This is a lookup from step names to producer classes.
# Only those step names are listed that are *initial producer steps*,
# i.e. ones that can form a producer (and may consume subsequent steps).
# For example, `repeat` is an initial producer step, but `times` is not,
# because it can only contribute to a `repeat` that has already been encountered.
all_known_initial_producer_steps = {
    CanonicalStepNames.V: V,
    CanonicalStepNames.E: E,
    CanonicalStepNames.addE: AddE,
    CanonicalStepNames.addV: AddV,
    CanonicalStepNames.and_: And,
    CanonicalStepNames.as_: As,
    CanonicalStepNames.barrier: Barrier,
    CanonicalStepNames.both: Both,
    CanonicalStepNames.bothE: BothE,
    CanonicalStepNames.bothV: BothV,
    CanonicalStepNames.cap: Cap,
    CanonicalStepNames.coalesce: Coalesce,
    CanonicalStepNames.constant: Constant,
    CanonicalStepNames.count: Count,
    CanonicalStepNames.drop: Drop,
    CanonicalStepNames.elementMap: ElementMap,
    CanonicalStepNames.emit: RepeatEmitUntil,
    CanonicalStepNames.flatMap: FlatMap,
    CanonicalStepNames.fold: Fold,
    CanonicalStepNames.has: Has,
    CanonicalStepNames.hasLabel: HasLabel,
    CanonicalStepNames.id_: Id,
    CanonicalStepNames.identity: Identity,
    CanonicalStepNames.in_: In,
    CanonicalStepNames.inE: InE,
    CanonicalStepNames.inV: InV,
    CanonicalStepNames.key: Key,
    CanonicalStepNames.label: Label,
    CanonicalStepNames.limit: Limit,
    CanonicalStepNames.none: NoneStep,
    CanonicalStepNames.not_: Not,
    CanonicalStepNames.or_: Or,
    CanonicalStepNames.order: OrderStep,
    CanonicalStepNames.otherV: OtherV,
    CanonicalStepNames.out: Out,
    CanonicalStepNames.outE: OutE,
    CanonicalStepNames.outV: OutV,
    CanonicalStepNames.path: Path,
    CanonicalStepNames.project: Project,
    CanonicalStepNames.properties: Properties,
    CanonicalStepNames.property_: Property,
    CanonicalStepNames.repeat: RepeatEmitUntil,
    CanonicalStepNames.select: Select,
    CanonicalStepNames.sideEffect: SideEffectStep,
    CanonicalStepNames.simplePath: SimplePath,
    CanonicalStepNames.store: Store,
    CanonicalStepNames.unfold: Unfold,
    CanonicalStepNames.union: Union,
    CanonicalStepNames.until: RepeatEmitUntil,
    CanonicalStepNames.value: Value,
    CanonicalStepNames.valueMap: ValueMap,
    CanonicalStepNames.values: Values,
    CanonicalStepNames.where: Where,
}


def bytecode_to_producer_chain(config: GremliteConfig, con: sqlite3.Connection, bytecode: Bytecode,
                               seed_producer: Optional[Producer] = None) -> Producer:
    """
    Scan a bytecode representation of a Gremlin traversal, chunk it into
    Producers, linked into a chain, and return the final Producer in the chain.
    """
    last_producer = seed_producer
    ss = StepStream(config, bytecode, existing_stream=seed_producer.ss if seed_producer else None)
    while ss.has_next():
        next_step = ss.peek_next()
        builder = all_known_initial_producer_steps.get(next_step.name)
        if builder is None:
            raise UnknownStep(next_step.name)  # pragma: no cover
        last_producer = builder(con, ss, parent=last_producer)
    return last_producer


@contextmanager
def carry_on(producer: Producer, bytecode: Bytecode, from_result=None, context=None):
    """
    Apply an ongoing traversal.

    :param producer: the Producer that wants to apply the ongoing traversal.
    :param bytecode: Bytecode representing the ongoing traversal.
    :param from_result: optional incoming Result. If not given, we use
        the producer's incoming result, or new.
    :param context: optional TraversalContext for the producer chain.
    :return: ProducerCursor
    """
    r = from_result or producer.incoming_result_or_new()
    seed = SingletonProducer(r, context=context)
    pro = bytecode_to_producer_chain(producer.config, producer.con, bytecode, seed_producer=seed)
    cur = ProducerCursor(pro)
    yield cur
    if not cur.stopped_iteration:
        # If and only if the cursor wasn't actually fully exhausted, we need to
        # do an abort here. When the cursor was *not* fully exhausted, this ensures
        # that any calls to `push_state()` on the `Result` object get their
        # corresponding `pop_state()`. If the cursor *was* fully exhausted, then
        # we must *not* call `abort()`, or else we'll cause *extra* calls to `pop_state()`,
        # which of course we do not want.
        pro.abort()
