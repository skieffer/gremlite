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

from abc import ABC
from collections import defaultdict
import copy
from enum import Enum
import sqlite3
from typing import List, Union

from gremlin_python.process.traversal import Bytecode, P
import gremlin_python.structure.graph as graph

from .cursors import StandinCursor, DummyCursor
from .errors import BadStepArgs, BadStepCombination
from .querytools import SubjectTypes
from .results import Result


class CanonicalStepNames:
    """
    These are the step names that appear in bytecode.

    They generally use camel case, not snake case, and omit the
    extra underscores that appear in certain step names in the Python bindings.

    To be clear: The *values* are as they appear in bytecode; the *names* used
    for them may still need extra underscores, because they are still Python names.
    """
    V = 'V'
    E = 'E'
    addE = 'addE'
    addV = 'addV'
    as_ = 'as'
    and_ = 'and'
    barrier = 'barrier'
    both = 'both'
    bothE = 'bothE'
    bothV = 'bothV'
    by = 'by'
    cap = 'cap'
    coalesce = 'coalesce'
    constant = 'constant'
    count = 'count'
    drop = 'drop'
    elementMap = 'elementMap'
    emit = 'emit'
    flatMap = 'flatMap'
    fold = 'fold'
    from_ = 'from'
    has = 'has'
    hasLabel = 'hasLabel'
    id_ = 'id'
    identity = 'identity'
    in_ = 'in'
    inE = 'inE'
    inV = 'inV'
    key = 'key'
    label = 'label'
    limit = 'limit'
    none = 'none'
    not_ = 'not'
    or_ = 'or'
    order = 'order'
    otherV = 'otherV'
    out = 'out'
    outE = 'outE'
    outV = 'outV'
    path = 'path'
    project = 'project'
    properties = 'properties'
    property_ = 'property'
    repeat = 'repeat'
    select = 'select'
    sideEffect = 'sideEffect'
    simplePath = 'simplePath'
    store = 'store'
    times = 'times'
    to = 'to'
    unfold = 'unfold'
    union = 'union'
    until = 'until'
    value = 'value'
    valueMap = 'valueMap'
    values = 'values'
    where = 'where'


filter_step_names = {
    CanonicalStepNames.has,
    CanonicalStepNames.hasLabel,
}

modulator_step_names = {
    CanonicalStepNames.by,
}

new_edge_modulator_step_names = {
    CanonicalStepNames.from_,
    CanonicalStepNames.to,
}

repeater_step_names = {
    CanonicalStepNames.emit,
    CanonicalStepNames.repeat,
    CanonicalStepNames.times,
    CanonicalStepNames.until,
}


class TraversalContext(Enum):
    BASIC = 'BASIC'
    PATTERN_MATCHING = 'PATTERN_MATCHING'


class GremliteConfig:
    """
    A class to hold configuration settings.
    """

    def __init__(self):
        ######################################################################################################
        # USER CONFIG
        #
        # Certain Gremlin steps, namely
        #   V(), E(), in(), out(), both(), in_e(), out_e(), both_e()
        # perform queries against the sqlite database which have the potential to return large sets
        # of results. These classes are therefore designed to open an `sqlite3.Cursor`, and *keep* it
        # open while the traversal is being processed, fetching only *one* result from it at a time.
        #
        # However, as demonstrated e.g. by our `tests.test_steps.test_locked_database()` test,
        # there can be issues if one `Connection` is holding a read cursor open for a long time,
        # while another `Connection` is trying to make changes. In case you are encountering
        # difficulties with this (which would be signalled by getting an `sqlite3.OperationalError`
        # saying, "database is locked" after the (default 5.0s) timeout), then you can set this
        # variable to `True`. This will cause the steps named above to instead fetch *all* results from
        # their cursors immediately, and then close those cursors.
        #
        # An issue of this kind is not expected to arise unless you are working with a busy, multithreaded
        # system, using multiple connections simultaneously. In particular, note that any single Gremlin
        # traversal is handled by a single Connection. This is why, for example, a traversal such as
        #   g.V().drop().iterate()
        # can work. The `V()` step holds a read cursor open while the `drop()` step is writing, but since
        # they are both using the same Connection, the database does not become locked.
        self.read_all_at_once = False

        # When a sequence of `has()` steps are used to filter results, it may be more efficient to
        # formulate some of these steps as SQLite queries, while performing others "on the Python side,"
        # i.e. simply retrieving object properties and filtering on these.
        #
        # Aiming to leave room open in the future for more sophisticated strategies, for now we simply
        # provide one called "basic heuristic filtering." By default it is switched on. You can turn it
        # off by setting this to `False`, and then *all* filter steps in a given sequence will be built
        # into one SQLite query; none will be saved to be performed on the Python side.
        self.use_basic_heuristic_filtering = True

        ######################################################################################################
        # TESTING CONFIG

        # This is purely for testing purposes. It causes the unit tests to include a case where we attempt
        # to use quad query constraint pattern [p|s]. That pattern is not supported, so the purpose of doing
        # this is to verify that our testing framework does indeed raise exceptions on unsupported patterns.
        self.use_p_bar_s_pattern = False

        # This is purely for testing purposes. It makes traversals iterate over `Result` objects, instead
        # of the Gremlin values those are meant to produce.
        self.traversal_returns_internal_result_iterator = False


class Step:
    """
    A class to represent Gremlin steps, based on their descriptions in bytecode.
    """

    def __init__(self, location, description):
        """
        :param location: string describing the location of the step in the traversal.
        :param description: a list, describing a single step, in bytecode.
        """
        self.loc = location
        self.name = description[0]
        self.args = description[1:]

    def __repr__(self):
        return f'Step {self.loc}: {self.name}({", ".join(repr(a) for a in self.args)})'

    def copy(self):
        return copy.deepcopy(self)

    def err_msg(self, base_msg):
        """Augment a given error message with a prefix indicating the step."""
        return f'{repr(self)}: {base_msg}'

    def extract_list_of_ids(self, accept_vertices=False, accept_edges=False):
        """
        Extract a list of IDs from our args.

        To be clear, args may be:
            * empty, or
            * any positive number of positive integers, or
            * any single list, tuple, or set of positive integers
        or the integers may also be vertices or edges depending on the kwargs passed (see below).

        :param accept_vertices: set True to also accept `graph.Vertex` instances besides integers
        :param accept_edges: set True to also accept `graph.Edge` instances besides integers
        :return: pair of lists: (ids, instances), where:
            ids: This is a list of all the IDs that were extracted. If graph objects
                were accepted, then their IDs appear here.
            instances: If graph objects were accepted, this is a list of all those that
                were found.
        """
        args = self.args

        given_ids = args
        if len(args) == 1:
            a = args[0]
            if isinstance(a, (list, tuple, set)):
                given_ids = a
            else:
                given_ids = [a]

        # Now `given_ids` is an iterable of the things that are supposed to give IDs.

        ids = []
        instances = []
        okay = True

        for g in given_ids:
            if isinstance(g, int) and g > 0:
                ids.append(g)
            elif accept_vertices and isinstance(g, graph.Vertex):
                ids.append(g.id)
                instances.append(g)
            elif accept_edges and isinstance(g, graph.Edge):
                ids.append(g.id)
                instances.append(g)
            else:
                okay = False
                break

        if not okay:
            opt = ' or graph objects' if accept_vertices or accept_edges else ''
            msg = self.err_msg(f"Args should be IDs{opt}, or an iterable thereof.")
            raise BadStepArgs(msg)

        return ids, instances

    def extract_string_varargs(self, allowed_numbers=None, nonzero=False, noun=None):
        return self.extract_varargs_of_types([str], allowed_numbers=allowed_numbers, nonzero=nonzero, noun=noun)

    def extract_bytecode_varargs(self, allowed_numbers=None, nonzero=False):
        noun = 'traversal' if allowed_numbers and list(allowed_numbers) == [1] else 'traversals'
        return self.extract_varargs_of_types([Bytecode], allowed_numbers=allowed_numbers, nonzero=nonzero, noun=noun)

    def check_property_value_arg(self, arg, noun=None, subject='Arg'):
        return self.check_arg(arg, [
            type(None), bool, int, float, str, P
        ], noun=noun, subject=subject)

    def check_arg(self, arg, types, noun=None, subject='Arg'):
        return self.extract_varargs_of_types(types, one_arg=arg, noun=noun, subject=subject)

    def extract_varargs_of_types(self, types, one_arg=None, allowed_numbers=None, nonzero=False,
                                 noun=None, subject='Args'):
        """
        Extract zero, one, or many args, all required to be in a given set of types.
        (An iterable of this type is *not* an accepted arg.)

        :param types: iterable of types that args are allowed to be
        :param one_arg: if not None, just check this one argument, instead of `self.args`.
        :param allowed_numbers: optional iterable of ints to require that the number of args
            lie in this set
        :param nonzero: set True to require a non-zero number of args.
        :param noun: optional indicator of what the arg(s) should be,
            such as "labels" or "property names", or singular "label" if exact_number is 1.
        :param subject: optional phrase to name the argument.
        :return: list of args (possibly empty)
        """
        args = [one_arg] if one_arg is not None else self.args
        types = tuple(types)
        if allowed_numbers is not None:
            allowed_numbers = list(allowed_numbers)
        if (
            not all(isinstance(a, types) for a in args) or
            (
                allowed_numbers is not None and len(args) not in allowed_numbers
            ) or
            (
                nonzero and len(args) == 0
            )
        ):
            np_parts = []
            if allowed_numbers is not None:
                n = len(allowed_numbers)
                if n == 1:
                    np_parts.append(str(allowed_numbers[0]))
                elif n == 2:
                    a, b = allowed_numbers
                    np_parts.append(f'{a} or {b}')
                else:
                    opts = ', '.join(allowed_numbers[:-1]) + f', or {allowed_numbers[-1]}'  # pragma: no cover
                    np_parts.append(opts)  # pragma: no cover
            elif nonzero:
                np_parts.append('one or more')
            if noun is None:
                if len(types) == 1:
                    noun = f'{types[0].__name__} instance{"" if allowed_numbers == [1] else "s"}'
                else:
                    noun = f'of the types: {{{", ".join(str(t) for t in types)}}}'
            np_parts.append(noun)
            noun_phrase = ' '.join(np_parts)
            raise BadStepArgs(self.err_msg(f"{subject} should be {noun_phrase}."))
        return args

    def check_no_args(self):
        """
        Check that we have no args.

        :return: nothing
        """
        if len(self.args) > 0:
            raise BadStepArgs(self.err_msg("Takes no args."))


class StepStream:
    """
    Stream class to help us consume the steps from a Bytecode instance.

    Also serves to store global data releevant to or accumulated during processing of a traversal.
    """

    def __init__(self, config: GremliteConfig, bytecode: Bytecode, existing_stream=None):
        self.bytecode = bytecode
        self.steps = [Step(str(i), d) for i, d in enumerate(self.bytecode.step_instructions)]
        self.N = len(self.steps)
        self.ptr = 0

        # Global data for the traversal:
        self.config = config
        self.storage_lists = existing_stream.storage_lists if existing_stream else defaultdict(list)

    def has_next(self):
        """Say whether we have another step"""
        return self.ptr < self.N

    def peek_next(self):
        """Return the next step, but do not consume it."""
        return self.steps[self.ptr]

    def take_next(self):
        """Consume the next step, and return it."""
        next_step = self.peek_next()
        self.ptr += 1
        return next_step

    def rewind(self, count=1):
        self.ptr = max(0, self.ptr - count)

    def insert_steps(self, steps: List[Step]):
        """Splice in a list of steps at the current pointer."""
        self.steps = self.steps[:self.ptr] + steps + self.steps[self.ptr:]
        self.N += len(steps)

    def take_contributing_steps(
            self,
            *step_names,
            allow_repeats=True,
            filter_steps=False,
            modulator_steps=False,
            new_edge_steps=False,
            repeater_steps=False,
    ):
        """
        Consume as many "contributing steps" as come up next.

        You can list individual step names and/or set the various kwargs to build
        the full set of steps you're willing to consume.

        :param step_names: list zero or more individual steps, using
            the `CanonicalStepNames` enum class
        :param allow_repeats: set False to accept each contributing step name at most once
        :param filter_steps: set True to accept {'has', 'has_label'}
        :param modulator_steps: set True to accept {'by'}
        :param new_edge_steps: set True to accept {'from', 'to'}
        :param repeater_steps: set True to accept {'emit', 'repeat', 'times', 'until'}

        :return: list of steps (possibly empty)
        """
        contributing_steps = ContributingSteps()

        matching_names = set(step_names)
        if filter_steps:
            matching_names.update(filter_step_names)
        if modulator_steps:
            matching_names.update(modulator_step_names)
        if new_edge_steps:
            matching_names.update(new_edge_modulator_step_names)
        if repeater_steps:
            matching_names.update(repeater_step_names)

        names_seen = set()

        def keep_going():
            if not self.has_next():
                return False
            name = self.peek_next().name
            is_okay = name in matching_names and (allow_repeats or name not in names_seen)
            if is_okay:
                names_seen.add(name)
            return is_okay

        while keep_going():
            contributing_steps.add_step(self.take_next())

        return contributing_steps


class ContributingSteps:
    """
    Container class to help manage collections of contributing steps, consumed
    after an initial step.
    """

    def __init__(self):
        self.all_steps = []

        self.filter_steps = []

        self.by_steps = []

        self.from_steps = []
        self.to_steps = []

        self.repeater_steps = []

    def add_step(self, step: Step):
        self.all_steps.append(step)
        name = step.name
        if name in filter_step_names:
            self.filter_steps.append(step)
        elif name == CanonicalStepNames.by:
            self.by_steps.append(step)
        elif name == CanonicalStepNames.from_:
            self.from_steps.append(step)
        elif name == CanonicalStepNames.to:
            self.to_steps.append(step)
        elif name in repeater_step_names:
            self.repeater_steps.append(step)

    @staticmethod
    def _extract_object_label(step: Step):
        """
        Extract the object label string that *should* have been passed as the one arg
        to certain types of steps, else raise an exception.

        :return: string
        """
        if len(step.args) != 1 or not isinstance(step.args[0], str):
            raise BadStepArgs(step.err_msg('requires one string arg'))
        return step.args[0]

    def _get_label_from_step_type(self, step_list):
        step = step_list[-1] if step_list else None
        return None if step is None else self._extract_object_label(step)

    def get_source(self):
        """Return the source label (str) named in a `from()` step, if any, else None."""
        return self._get_label_from_step_type(self.from_steps)

    def get_target(self):
        """Return the target label (str) named in a `to()` step, if any, else None."""
        return self._get_label_from_step_type(self.to_steps)


class Producer(ABC):
    """
    Generic base class for all producers.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None, context=None):
        """
        :param con: connection to sqlite database.
        :param ss: the step stream from which we should consume steps, in order to
            build our plan.
        :param parent: our parent Producer, if any.
        :param context: optional TraversalContext, to be inherited by all child producers,
            unless they override it.
        """
        self.con = con
        self.ss = ss
        self.parent = parent
        self._context = context

        self.cursor = None
        self.current_incoming_result = None

        self.base_step = ss.take_next()
        self.c_steps = ContributingSteps()
        self.build(ss)

    @property
    def config(self):
        return self.ss.config

    @property
    def context(self):
        if self._context is not None:
            return self._context
        elif self.parent is not None:
            return self.parent.context
        else:
            return TraversalContext.BASIC

    @property
    def is_root_or_barrier(self):
        return self.parent is None

    def build_err_msg(self, base_msg):
        """Augment a given error message with a prefix indicating the step in which it occurred."""
        return self.base_step.err_msg(base_msg)

    def __next__(self):
        """
        Serve as an iterator for the gremlin traversal steps like `next()`,
        `iterate()`, `to_list()`, `to_set()`, that retrieve graph results.
        """
        r = self.get_next_result()
        return r.gremlin_value

    def __iter__(self):
        return ProducerInternalResultIterator(self)

    def empty_all_cursors_in_chain(self):
        self.cursor = DummyCursor.emtpy_cursor()
        if self.parent:
            self.parent.empty_all_cursors_in_chain()

    def abort(self):
        self.empty_all_cursors_in_chain()
        try:
            self.get_next_result()
        except StopIteration:
            pass

    def get_next_result(self) -> Result:
        # If we have a cursor at this point, then it has already produced at least one result.
        # So it makes sense now to pop the state of the incoming result. This also ensures that
        # our parent producer gets to work with the result in the state it expects.
        if self.cursor is not None:
            if self.in_r:
                self.in_r.pop_state()
        # Cursors do not raise StopIteration when they're exhausted,
        # they just return None. So, by doing this...
        next_cursor_item = None if self.cursor is None else self.cursor.fetchone()
        # ...we can say that `next_cursor_item` is None if and only if we're ready
        # to try to get a new cursor.
        while next_cursor_item is None:
            if self.is_root_or_barrier:
                if self.cursor is not None:
                    # It's a root or barrier producer which has already made its one cursor,
                    # and the cursor is exhausted. This is how iteration stops.
                    raise StopIteration
            else:
                # We're only meant to make one cursor per incoming value from our
                # parent producer, so now is the time to get our parent's next value.
                #
                # This may raise a `StopIteration`. We don't catch it. That's because
                # if we have a parent producer and it is all out of results, then we
                # are not meant to produce anything more, or have side-effects, ourselves.
                #
                # This is true even of steps like `V()` and `addV()`. If those steps have
                # no parent at all, then they do something; but if they *have* a parent,
                # which returns no results, then they are meant to do nothing (I verified
                # this against TinkerGraph 3.7.2, on 240930).
                self.current_incoming_result = self.parent.get_next_result()
                self.check_incoming_result_type()
            self.cursor = self.make_next_cursor()
            # Even though we have a new cursor, it may not produce any results, i.e. the
            # next call may produce None. However, we may still be able to get another
            # input from our parent producer, and try again. That's why we're in a while loop.
            next_cursor_item = self.cursor.fetchone()
        if self.in_r:
            self.in_r.push_state()
        return self.assemble_outgoing_result(next_cursor_item)

    @property
    def in_r(self) -> Result:
        return self.current_incoming_result

    def incoming_result_or_new(self) -> Result:
        return Result(self.con, self.ss) if self.in_r is None else self.in_r

    ##########################################################################################
    # Subclasses MAY implement

    def check_incoming_result_type(self):
        """
        Subclasses may check the type of the incoming result, and raise an exception
        if it is not as required.
        """
        pass

    def assemble_outgoing_result(self, next_cursor_item) -> Result:
        """
        Given the next item returned by our current cursor, assemble the
        `Result` we want to pass on to the next producer.

        Since our cursors may produce various types of items (database rows for sqlite cursors,
        or anything else for StandinCursors), but our `get_next_result()` method has to always
        return a `Result`, this is a chance to transform the incoming items.

        Often, this method should augment the incoming `Result` object in some way, and return
        that. It can rely on the fact that `push_state()` has already been called on that object,
        as a part of our `get_next_result()` method, just prior to invocation of this method.

        :param next_cursor_item: the next item from our current cursor
        :return: a Result object
        """
        return next_cursor_item

    ##########################################################################################
    # Subclasses MUST implement

    def build(self, ss: StepStream):
        """
        Build ourselves, i.e. construct our plan, based on the given step stream.

        When this method is invoked, `self.base_step` is *already* equal to the step that defines
        us, and that step has already been consumed from the step stream. We now have an opportunity
        to consume any *additional* steps that we may be able to accept (such as an `as()` modulator).

        :param ss: StepStream
        :return: nothing
        """
        raise NotImplementedError  # pragma: no cover

    def make_next_cursor(self) -> Union[sqlite3.Cursor, StandinCursor]:
        """
        Make our next cursor. This can mean performing an actual query to the underlying sqlite
        database, producing an `sqlite3.Cursor`, or it can mean assembling one of our other cursor
        classes, to iterate over some other kind of objects.

        Should assume that, if we have a parent producer, and if we need to use its current
        result to formulate our query, then its result (a) is already in `self.current_incoming_result`,
        and (b) has already been checked by `self.check_incoming_result_type()`.

        :return: either an sqlite3.Cursor object that's ready to iterate over the rows of a
            new query, or a StandinCursor ready to iterate over something else (which our
            `assemble_outgoing_result()` method is ready to receive).
        """
        raise NotImplementedError  # pragma: no cover


class ProducerCursor(StandinCursor):
    """
    Act like a cursor, but return the Results from a Producer's get_next_result() method.
    """

    def __init__(self, producer: Producer):
        self.producer = producer
        self.previewed_result = None
        self.stopped_iteration = False

    def has_next(self):
        if self.previewed_result is None:
            self.previewed_result = self.fetchone()
        return self.previewed_result is not None

    def fetchone(self):
        if self.previewed_result is not None:
            tmp = self.previewed_result
            self.previewed_result = None
            return tmp
        else:
            try:
                result = self.producer.get_next_result()
            except StopIteration:
                result = None
                self.stopped_iteration = True
            return result


class SingletonProducer(Producer):
    """
    A producer pre-laoded to produce exactly one result and then stop.

    Useful for steps that want to take a current Result and feed it into an ongoing
    traversal that was passed as an argument (using the `__` object in gremlinpython).
    E.g. a `select()` step that wants to apply further bytecode from a `by()` modulator.
    """

    def __init__(self, result: Result, context=None):
        bc = Bytecode()
        bc.add_step('_singleton_producer_dummy_step_')
        ss = StepStream(result.ss.config, bc, existing_stream=result.ss)
        super().__init__(result.con, ss, context=context)
        self.singleton_result = result

    def build(self, ss: StepStream):
        pass

    def make_next_cursor(self) -> Union[sqlite3.Cursor, StandinCursor]:
        return DummyCursor([self.singleton_result])


class ProducerInternalResultIterator:
    """
    As an iterator, the `Producer` class iterates over *Gremlin* results.
    If you want to instead iterate over its *internal* `Result` objects,
    wrap it in this class.
    """

    def __init__(self, producer: Producer):
        self.producer = producer

    def __next__(self):
        return self.producer.get_next_result()


class ConsumerProducer(Producer, ABC):
    """
    A producer that requires a certain type of incoming result.
    """
    
    @property
    def required_type(self):
        """
        Subclasses MUST override.
        
        :return: the type or tuple of types that the incoming result is required to be.
        """
        raise NotImplementedError  # pragma: no cover
    
    def additional_checks(self):
        """
        Subclasses MAY override. This is a chance to perform any additional checks,
        or record any special information.
        
        :return: nothing
        """
        pass

    def check_incoming_result_type(self, err_msg_extension=''):
        r = self.in_r
        if r is None or not isinstance(r.last_object, self.required_type):
            received = 'nothing' if r is None else str(r.last_object)
            msg = f'received {received} but'
            msg += f' requires incoming {self.required_type}{err_msg_extension}'
            raise BadStepCombination(self.build_err_msg(msg))
        self.additional_checks()


class ElementConsumerProducer(ConsumerProducer, ABC):
    """
    A producer that requires incoming result to be a vertex or an edge,
    or optionally vertex or edge properties (if set by subclass).
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        # Subclasses may set these booleans to allow properties:
        self.accept_vertex_properties = False
        self.accept_edge_properties = False
        # id valence is 1 for vertices, -1 for edges, and 0 for properties:
        self.incoming_result_id_valence = None
        self.incoming_subject_type = None
        super().__init__(con, ss, parent=parent)

    @property
    def required_type(self):
        rt = [graph.Vertex, graph.Edge]
        if self.accept_vertex_properties:
            rt.append(graph.VertexProperty)
        if self.accept_edge_properties:
            rt.append(graph.Property)
        return tuple(rt)

    def additional_checks(self):
        r = self.in_r
        if r.is_vertex():
            self.incoming_result_id_valence = 1
            self.incoming_subject_type = SubjectTypes.VERTICES
        elif r.is_edge():
            self.incoming_result_id_valence = -1
            self.incoming_subject_type = SubjectTypes.EDGES
        elif isinstance(r.last_object, graph.VertexProperty):
            self.incoming_result_id_valence = 0
            self.incoming_subject_type = SubjectTypes.VERTICES
        elif isinstance(r.last_object, graph.Property):
            self.incoming_result_id_valence = 0
            self.incoming_subject_type = SubjectTypes.EDGES


class VertexConsumerProducer(ConsumerProducer, ABC):
    """
    A producer that requires incoming result to be a vertex.
    """

    @property
    def required_type(self):
        return graph.Vertex


class EdgeConsumerProducer(ConsumerProducer, ABC):
    """
    A producer that requires incoming result to be an edge.
    """

    @property
    def required_type(self):
        return graph.Edge


class PropertyConsumerProducer(ConsumerProducer, ABC):
    """
    A producer that requires incoming result to be a property.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.incoming_subject_type = None
        super().__init__(con, ss, parent=parent)

    @property
    def required_type(self):
        return graph.VertexProperty, graph.Property

    def additional_checks(self):
        r = self.in_r
        obj = r.last_object
        if isinstance(obj, graph.VertexProperty):
            self.incoming_subject_type = SubjectTypes.VERTICES
        elif isinstance(obj, graph.Property):
            self.incoming_subject_type = SubjectTypes.EDGES


class ListConsumerProducer(ConsumerProducer, ABC):
    """
    A producer that requires incoming result to be a list.
    """

    @property
    def required_type(self):
        return list


class SideEffect(ElementConsumerProducer, ABC):
    """
    A producer that needs an input, but just performs a side-effect, and then either
    forwards the incoming object onward as its only output, or else returns no output at all.
    """

    def side_effect(self):
        """
        Subclasses must implement.

        :return: nothing
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def do_forward(self):
        """
        Subclasses MAY override.

        :return: boolean, saying whether the incoming object should be forwarded.
        """
        return True

    def make_next_cursor(self):
        self.side_effect()
        if self.do_forward:
            return DummyCursor([self.in_r])
        else:
            return DummyCursor.emtpy_cursor()


class EvaluatorProducer(Producer, ABC):
    """
    A producer that moves to a non-graph value.
    """

    def make_next_cursor(self):
        try:
            if self.is_multi:
                values = self.evaluate(self.in_r)
            else:
                value = self.evaluate(self.in_r)
                values = [value]
        except StopIteration:
            return DummyCursor.emtpy_cursor()
        return DummyCursor(values)

    @property
    def is_multi(self):
        """
        Subclasses may override and return True to signal that instead of
        producing a single value, their evaluate() method produces a list of
        values, over which our cursor should iterate one at a time.

        :return: boolean
        """
        return False

    def evaluate(self, r: Result):
        """
        Subclasses must implement

        :param r: the current incoming Result object.
        :return: the non-graph value to which to move.
        :raises: StopIteration to signal that instead of moving to a non-graph value,
            we actually want to produce no result.
        """
        raise NotImplementedError  # pragma: no cover

    def assemble_outgoing_result(self, next_cursor_item):
        r = self.incoming_result_or_new()
        value = next_cursor_item
        r.move_to_value(value)
        return r


class BarrierStep(Producer, ABC):
    """
    A barrier step, i.e. one which first consumes all results coming before it, and
    then produces something based on that.

    Some barrier steps produce exactly one thing, and move us to a `Result` having
    only that one thing in its path. We call these "restart barriers."
    See `RestartBarrier` subclass.

    Other barrier steps produce either all the same Result objects that they received,
    or else some subset (possibly re-ordered) thereof. We call these "pass-through
    barriers." See `PassthroughBarrier` subclass.
    """

    @property
    def is_root_or_barrier(self):
        return True

    def make_next_cursor(self) -> Union[sqlite3.Cursor, StandinCursor]:
        for r in self.parent:
            self.for_each_incoming_result(r)
        return DummyCursor(self.make_results_list())

    def for_each_incoming_result(self, r: Result):
        """
        Subclasses MAY implement.

        Will be invoked once for each result generated by all the foregoing
        steps of the traversal. Subclasses probably want to build something
        they initialized in their `__init__()` method.

        :param r: the next Result
        :return: nothing
        """
        pass

    def make_results_list(self) -> List[Result]:
        """
        Subclasses MUST implement.

        This is a chance to do some finalizing steps, to complete the thing that
        was being built during the calls to our `for_each_incoming_result()` method.

        :return: the list of Result objects this step is to produce
        """
        raise NotImplementedError  # pragma: no cover


class RestartBarrier(BarrierStep, ABC):
    """
    A barrier step that produces exactly one thing, and moves us to a Result having
    only that one thing in its path.
    """

    def make_results_list(self):
        r = Result(self.con, self.ss)
        r.move_to_value(self.final_value)
        return [r]

    @property
    def final_value(self):
        """
        Subclasses MUST implement.

        :return: the object that is to be the one object in the path of the one Result
            this barrier step produces.
        """
        raise NotImplementedError  # pragma: no cover


class PassthroughBarrier(BarrierStep, ABC):
    """
    A barrier step that produces all, or a subset of, the incoming Results it receives.
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.all_incoming_results = []
        super().__init__(con, ss, parent=parent)

    def for_each_incoming_result(self, r: Result):
        # Note: Use of `mixed_copy()` here is experimental.
        # If we're getting issues, then maybe just need to use `r.copy()` instead.
        # The hope is that `mixed_copy()` is faster (and uses less space).
        self.all_incoming_results.append(r.mixed_copy())

    def make_results_list(self):
        """
        Subclasses MAY override.

        :return: list of Result objects to be produced by this step.
        """
        return self.all_incoming_results


class ElementEvaluatorProducer(EvaluatorProducer, ElementConsumerProducer, ABC):
    """
    An evaluator producer that needs to find an incoming element.
    """


class FilterProducer(Producer, ABC):
    """
    A step that either allows the incoming result to pass, or does not.
    """

    def passes(self, r: Result):
        """
        Subclasses MUST implement

        :param r: the current incoming Result object
        :return: True if the incoming object passes, False if not
        """
        raise NotImplementedError  # pragma: no cover

    def make_next_cursor(self):
        if self.passes(self.in_r):
            return DummyCursor.one_time_cursor()
        return DummyCursor.emtpy_cursor()

    def assemble_outgoing_result(self, next_cursor_item) -> Result:
        return self.incoming_result_or_new()


class ElementFilterProducer(FilterProducer, ElementConsumerProducer, ABC):
    """
    A filter step that requires incoming vertices or edges.
    """


class ModulatingProducer(Producer, ABC):
    """
    A Producer that needs to set up a cycle of modulators as determined
    by contributing `by()` steps. These modulators map objects either:
        * to themselves (`by()` with no args)
        * to one of their property values (`by(prop_name)`)
        * to the first result of an ongoing traversal (`by(traversal)`)
    """

    def __init__(self, con: sqlite3.Connection, ss: StepStream, parent=None):
        self.modulators = None
        self.num_modulators = None
        self.mod_ptr = None
        super().__init__(con, ss, parent=parent)

    def build(self, ss: StepStream):
        self.special_build(ss)

        self.c_steps = ss.take_contributing_steps(*self.c_steps_varargs, modulator_steps=True, **self.c_steps_kwargs)
        # The "modulators" will be a list of the arguments that were passed to any `by()` steps,
        # or `None` for `by()` steps that were given no arg.
        self.modulators = []
        for step in self.c_steps.by_steps:
            step = self.preprocess_by_step(step)
            args = step.extract_varargs_of_types([str, Bytecode],
                                                 allowed_numbers=[0, 1], noun='property name or traversal')
            modulator = args[0] if args else None

            # Reduce prop names to traversals.
            if isinstance(modulator, str):
                # The string is to be interpreted as a property name, and our job is
                # to obtain the property *value* (not property *object*).
                #
                # Note: TinkerGraph has a behavior here which I think is dumb, and I'm
                # deliberately diverging from it. Namely, when there are multiple property
                # values, it raises an exception and advises you to use a `__.properties(prop_name)`
                # traversal instead. But doing so will return the first property *object* (not *value*).
                #
                # If you try to use `__.values(prop_name)` it raises the same exception, but if
                # you use `__.properties(prop_name).value()` (which is equivalent), finally it
                # allows that to work.
                #
                # Why not just reduce this case to `__.values(prop_name)` for you, and let that
                # actually work, and return the first property *value*? That's what we're doing.
                prop_name = modulator
                modulator = Bytecode()
                modulator.add_step('values', prop_name)

            self.modulators.append(modulator)
        # We always want at least one modulator. `None` means, "Keep the object as it is."
        if len(self.modulators) == 0:
            self.modulators.append(None)

        self.num_modulators = len(self.modulators)
        self.mod_ptr = 0

    def next_modulator(self) -> Union[Bytecode, None]:
        """Cycle infinitely through the modulator list"""
        modulator = self.modulators[self.mod_ptr]
        self.mod_ptr = (self.mod_ptr + 1) % self.num_modulators
        return modulator

    ##############################################################################
    # Subclasses MAY implement

    @property
    def c_steps_varargs(self):
        """
        Subclasses may implement.

        :return: a list of varargs you need to add when we take contributing steps
        from the step stream during build.
        """
        return []

    @property
    def c_steps_kwargs(self):
        """
        Subclasses may implement.

        :return: a dict of kargs you need to add when we take contributing steps
        from the step stream during build.
        """
        return {}

    def preprocess_by_step(self, step: Step) -> Step:
        """
        Subclasses may implement.

        This is a chance for subclasses to play with the args passed to by() steps,
        for example removing special ones and returning a copy of the step, with
        special args deleted.

        :param step: the step as given by the user
        :return: the same step or a modified version
        """
        return step

    def special_build(self, ss: StepStream):
        """
        Subclasses may implement.

        This is a chance to do any additional build steps required by the subclass,
        besides setting up the modulator cycle.

        Subclasses should NOT call `ss.take_contributing_steps()`. Instead, if necessary,
        they can override the `c_steps_varargs()` and/or `c_steps_kwargs()` property methods
        in order to include more types of contributing steps besides those already included
        `modulator_steps=True`.

        :param ss: StepStream
        :return: nothing
        """
        pass


class ModulatingEvaluatorProducer(ModulatingProducer, EvaluatorProducer, ABC):
    """
    Modulating producer that's also an evaluator.
    """


class ModulatingElementEvaluatorProducer(ModulatingProducer, ElementEvaluatorProducer, ABC):
    """
    Modulating producer that's also an element evaluator.
    """
