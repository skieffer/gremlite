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

import sqlite3
from typing import List, Optional

from gremlin_python.process.traversal import P

from .base import Step, CanonicalStepNames, GremliteConfig
from .errors import BadStepArgs, BadStepCombination
from .results import Result
import gremlite.querytools as qt


def heuristic_filter_split(
        incoming_steps: List[Step], initial_query_is_broad=False) -> (List[Step], List[Step]):
    """
    Divide a list of `has()` steps into those that should be applied on the query side,
    and those that should be applied on the element side.

    Our strategy is simple: In almost all cases, steps involving text predicates are kept
    on the element side, while all others are put on the query side.

    The reason for this is that text predicate filters are thought to be potentially very costly
    to apply on the query side. Consider for example the following step:

        has('name', TextP.lt('Foo'))

    In order to apply this filter on the query side, we have to constrain the `o` column to lie
    within the set of all integer values that happen to encode strings that are lexicographically
    less than 'Foo'. This could be a huge set, and it is not an interval but a lawless set of
    scattered integers (because strings are added to the strings table not in lexicographic order,
    but in the order they first appear in the database). Depending on the size of the database,
    this could potentially be a costly query to carry out.

    Compare this to what it means to instead apply a text predicate "on the element side." This
    means that, if say we are filtering vertices, we instead grab the actual value of the 'name'
    property for each incoming vertex, and check whether that value is < 'Foo'. If there is a
    small number of incoming vertices, this seems like a far better approach.

    The one exception is that if the initial query is thought to be "broad" (see below), *and*
    if *all* of the given filter steps involve text predicates, then the first filter step is put
    on the query side, while all others remain on the element side.

    A "broad" initial query is one that is thought to do such a poor job of narrowing down,
    that we are actually better off applying a text predicate on the query side than not.
    For example, in `V(2, 3, 5)` the initial query selects exactly three vertices, and that
    is a very narrow initial query; but in `V()` the initial query is that `s > 0` which is
    very broad.
    """
    if not incoming_steps:
        return [], []

    non_text_pred_steps, text_pred_steps = [], []
    for step in incoming_steps:
        if any(qt.is_text_predicate(a) for a in step.args):
            text_pred_steps.append(step)
        else:
            non_text_pred_steps.append(step)

    if initial_query_is_broad and not non_text_pred_steps:
        query_steps, element_steps = text_pred_steps[:1], text_pred_steps[1:]
    else:
        query_steps, element_steps = non_text_pred_steps, text_pred_steps

    return query_steps, element_steps


class Filterable:
    """
    Represents a thing to be filtered.
    """
    ...


class FilterableQuery(Filterable):

    def __init__(self, subject_type: qt.SubjectTypes, s_subquery: str):
        """
        :param subject_type: tells what type of subject we are filtering
        :param s_subquery: optional subquery on the s column. Can be anything that
            could be passed to the `s` kwarg of the `querytools.write_quads_query()` function.
        """
        self.subject_type = subject_type
        self.s_subquery = s_subquery


class FilterableElement(Filterable):

    def __init__(self, result: Optional[Result]):
        """
        :param result: a Result, or None
        """
        self.result = result


class FilterHandler:

    def dispatch(self, filter_step: Step, filterable: Filterable) -> Filterable:
        args = filter_step.args

        if filter_step.name == CanonicalStepNames.has:
            n = len(args)
            if n == 1:
                # has(key)
                key = args[0]
                filter_step.check_arg(key, [str], subject='key arg')
                return self.unary_has(filter_step, key, filterable)

            elif n == 2:
                # has(key, value)
                key, value = args

                filter_step.check_arg(key, [str], subject='key arg')
                filter_step.check_property_value_arg(value, subject='value arg')

                return self.binary_has(filter_step, key, value, filterable)

            elif n == 3:
                # has(label, key, value)
                label, key, value = args

                filter_step.check_arg(label, [str], subject='label arg')
                filter_step.check_arg(key, [str], subject='key arg')
                filter_step.check_property_value_arg(value, subject='value arg')

                filterable = self.dispatch(
                    Step(f'{filter_step.loc} (part 1)', [CanonicalStepNames.hasLabel, label]),
                    filterable
                )
                return self.dispatch(
                    Step(f'{filter_step.loc} (part 2)', [CanonicalStepNames.has, key, value]),
                    filterable
                )

            else:
                raise BadStepArgs(filter_step.err_msg('Wrong number of args'))

        elif filter_step.name == CanonicalStepNames.hasLabel:
            # has_label(*labels)
            labels = args

            if len(labels) == 1:
                filter_step.check_arg(labels[0], [str, P], subject='Label arg')
            else:
                filter_step.extract_string_varargs(nonzero=True)

            return self.has_label(filter_step, labels, filterable)

        else:
            raise BadStepCombination(filter_step.err_msg('Unknown filter step'))  # pragma: no cover

    def unary_has(self, filter_step: Step, key: str, filterable: Filterable) -> Filterable:
        raise NotImplementedError  # pragma: no cover

    def binary_has(self, filter_step: Step, key: str, value, filterable: Filterable) -> Filterable:
        raise NotImplementedError  # pragma: no cover

    def has_label(self, filter_step: Step, labels, filterable: Filterable) -> Filterable:
        raise NotImplementedError  # pragma: no cover


class KnownEmptyResult(Exception):
    """Query is guaranteed to be empty."""
    ...


class FilterQueryBuilder(FilterHandler):
    """
    Builds an SQLite query string, based on a given list of `has()` steps.
    """

    def __init__(self, config: GremliteConfig, con: sqlite3.Connection, filter_steps: List[Step]):
        """
        :param con: connection
        :param filter_steps: list of `has()` steps
        """
        self.config = config
        self.con = con
        self.filter_steps = filter_steps

    def build(self, filterable: FilterableQuery) -> str:
        """Build the SQLite query string. """
        for step in self.filter_steps:
            try:
                filterable = self.dispatch(step, filterable)
            except KnownEmptyResult:
                return qt.write_quads_query(sel='s', s=qt.Constants.IMPOSSIBLE_SUBJECT)
        return filterable.s_subquery

    def unary_has(self, filter_step: Step, key: str, filterable: FilterableQuery) -> FilterableQuery:
        subject_type = filterable.subject_type
        s_subquery = filterable.s_subquery

        s = None
        p = qt.encode_property_name(self.con, key, merge=False)
        g = None

        if p is None:
            # This means the property name doesn't exist yet at all, so there are certainly no quads that have it.
            raise KnownEmptyResult
        else:
            # Since o is unconstrained, we cannot use our (ops) index. Therefore we
            # have to be careful not to formulate a [p|s] constraint pattern, since we have no (ps) index.
            # If s is exact, we can use our (sp) index, but if it is not, then we must
            # use our (pg) index instead.
            if self.config.use_p_bar_s_pattern:
                # This is purely for testing purposes, to verify that our testing framework does indeed raise
                # exceptions on unsupported patterns.
                s = s_subquery
            elif s_subquery is None or isinstance(s_subquery, qt.Sign):
                # The s_subquery is inexact, but we can incorporate the subject type using
                # the g column instead. We'll have a [pg|] pattern for edge properties, and [p|g] for
                # vertex properties, and in either case will use index (pg).
                g = qt.Constants.NON_ID_G_VALUE if subject_type is qt.SubjectTypes.EDGES else qt.Sign.POS
            else:
                # The s_subquery is exact, so we have an [sp|] pattern and will use index (sp).
                s = s_subquery

        # We select 'DISTINCT s' in order to filter out the repeats that would otherwise arise if the
        # requested property p is present in plural cardinality.
        query = qt.write_quads_query(sel='DISTINCT s', s=s, p=p, g=g)
        return FilterableQuery(subject_type, query)

    def binary_has(self, filter_step: Step, key: str, value, filterable: FilterableQuery) -> FilterableQuery:
        subject_type = filterable.subject_type
        s_subquery = filterable.s_subquery

        p = qt.encode_property_name(self.con, key, merge=False)
        o = qt.encode_property_value(self.con, value, merge=False)

        if p is None or o is None:
            # This means the property name or value doesn't exist yet at all,
            # so there are certainly no quads that have this combination.
            raise KnownEmptyResult
        else:
            s = qt.Sign.valence(subject_type) if s_subquery is None else s_subquery

        query = qt.write_quads_query(sel='s', s=s, p=p, o=o)
        return FilterableQuery(subject_type, query)

    def has_label(self, filter_step: Step, labels, filterable: FilterableQuery) -> FilterableQuery:
        subject_type = filterable.subject_type
        s_subquery = filterable.s_subquery

        label_constraints = [qt.encode_label(self.con, subject_type, label, merge=False) for label in labels]
        label_constraints = [i for i in label_constraints if i is not None]

        n = len(label_constraints)

        if n == 0:
            # None of the requested labels exists yet, so there certainly aren't any objects having them.
            raise KnownEmptyResult

        label_constraint = label_constraints[0] if n == 1 else label_constraints

        if subject_type == qt.SubjectTypes.VERTICES:
            query = qt.write_quads_query(
                sel='s', s=s_subquery, p=qt.Constants.HAS_LABEL_PREDICATE, o=label_constraint
            )
        elif subject_type == qt.SubjectTypes.EDGES:
            query = qt.write_quads_query(
                sel='g', p=label_constraint, g=s_subquery
            )
        else:
            raise BadStepArgs(filter_step.err_msg('Unknown subject type'))  # pragma: no cover

        return FilterableQuery(subject_type, query)
