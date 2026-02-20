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

from collections import defaultdict
import copy
import sqlite3

from gremlin_python.process.traversal import Cardinality, Direction, T, Traverser
import gremlin_python.structure.graph as graph

from .errors import BadStepCombination, UnsupportedUsage
import gremlite.querytools as qt


class RestorableDictionary:
    """
    A dictionary which keeps track of changes, so that they can be rolled back.
    """

    def __init__(self):
        self.d = {}
        self.stack = []

    def mixed_copy(self):
        d = RestorableDictionary()
        d.d = self.d.copy()
        d.stack = self.stack[:]
        if d.stack:
            # Only the last item might be modified by both us and our new copy,
            # so make a deep copy of that one.
            d.stack[-1] = copy.deepcopy(d.stack[-1])
        return d

    def __setitem__(self, key, value):
        if self.stack:
            new_keys, old_pairs = self.stack[-1]
            if key not in self.d:
                new_keys.add(key)
            elif key not in old_pairs:
                old_pairs[key] = self.d[key]
        self.d[key] = value

    def clear(self):
        if self.stack:
            _, old_pairs = self.stack[-1]
            for key in self.d:
                if key not in old_pairs:
                    old_pairs[key] = self.d[key]
        self.d.clear()

    def get(self, key, default=None):
        return self.d.get(key, default)

    def push(self):
        new_keys = set()
        old_pairs = {}
        self.stack.append((new_keys, old_pairs))

    def pop(self):
        if self.stack:
            new_keys, old_pairs = self.stack.pop()
            for key in new_keys:
                del self.d[key]
            self.d.update(old_pairs)


class FakeTraverser:
    """
    This allows an object to be returned as a traverser.
    """

    @property
    def object(self):
        return self

    @property
    def bulk(self):
        return 1

    @bulk.setter
    def bulk(self, value):
        pass


class Result(FakeTraverser):
    """
    The Result class is what we use to keep track of the path and all other data, as
    we conduct a traversal.

    Note: We make it a subclass of `FakeTraverser` purely for testing purposes.
    This allows the `traversal_returns_internal_result_iterator` config switch to work.
    """

    def __init__(self, con: sqlite3.Connection, ss):
        """
        :param con: the connection
        :param ss: the StepStream for the traversal being processed
        """
        self.con = con
        self.ss = ss

        self.vertices_by_id = {}
        self.edges_by_id = {}

        self.N = 0
        self.objects = []
        self.labels = []
        self.labeled_objects = RestorableDictionary()

        self.stack = []

        # A place for the `order()` step to stash sort keys:
        self._order_keys = []

    def mixed_copy(self):
        """Some data members are shared, for others we get a new copy, but not a deep one."""
        r = Result(self.con, self.ss)

        r.vertices_by_id = self.vertices_by_id
        r.edges_by_id = self.edges_by_id

        r.N = self.N
        r.objects = self.objects[:]
        r.labels = self.labels[:]
        r.labeled_objects = self.labeled_objects.mixed_copy()

        r.stack = self.stack[:]

        return r

    def push_state(self):
        self.stack.append(len(self))
        self.labeled_objects.push()

    def pop_state(self):
        if self.stack:
            n = self.stack.pop()
            # On truncating a list: https://stackoverflow.com/a/4838541
            del self.objects[n:]
            del self.labels[n:]
            self.N = n
            self.labeled_objects.pop()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # This is so that `deepcopy()` can be applied in our `copy()` method.
        if 'con' in state:
            del state['con']
        return state

    def __len__(self):
        return self.N

    @property
    def last_object(self):
        return self.objects[-1] if self.objects else None

    @property
    def penultimate_object(self):
        return self.objects[-2] if len(self) >= 2 else None

    @property
    def last_subject_type(self):
        return self.get_subject_type(self.last_object)

    @property
    def last_object_sort_key(self):
        """
        :return: the sort key by which our last object should be compared to other objects
        """
        obj = self.last_object
        if isinstance(obj, (graph.Vertex, graph.Edge, graph.VertexProperty)):
            return obj.id
        elif isinstance(obj, graph.Property):
            return obj.value
        else:
            return obj

    @property
    def order_keys(self):
        return self._order_keys

    def clear_order_keys(self):
        self.order_keys.clear()

    def add_order_key(self, key):
        self.order_keys.append(key)

    def add_to_storage_list(self, list_name, obj):
        self.ss.storage_lists[list_name].append(obj)

    def get_storage_list(self, list_name):
        return self.ss.storage_lists[list_name]

    @staticmethod
    def get_subject_type(obj):
        if isinstance(obj, graph.Vertex):
            return qt.SubjectTypes.VERTICES
        elif isinstance(obj, graph.Edge):
            return qt.SubjectTypes.EDGES
        return None

    @property
    def gremlin_value(self) -> Traverser:
        obj = self.last_object
        # We wait to the last moment to bother hydrating objects:
        self.hydrate_object(obj)
        return Traverser(obj)

    def is_vertex(self, empty_okay=False):
        obj = self.last_object
        return (
            isinstance(obj, graph.Vertex) or
            (
                empty_okay and obj is None
            )
        )

    def is_edge(self, empty_okay=False):
        obj = self.last_object
        return (
            isinstance(obj, graph.Edge) or
            (
                empty_okay and obj is None
            )
        )

    def copy(self, stop_at_latest=None, rewind=None):
        """
        Make a copy of this Result.

        :param stop_at_latest: optional object label (string). If given, we will truncate the
            object path to the initial sequence ending at (and including) the
            *last* object that was assigned this label. (I.e. if multiple objects were
            assigned the same label, we will go up to the latest one.) If the given label
            was never assigned, you will get an empty path.

        :param rewind: pass a positive integer in order to discard this many steps off
            the (most recent) end of the object sequence.

        :return: Result
        """
        r = copy.deepcopy(self)
        r.con = self.con

        # We want the *same* (not a copy) StepStream, so that its global records are preserved.
        r.ss = self.ss

        total_rewind = 0

        if stop_at_latest is not None:
            obj_label = stop_at_latest

            k = -1
            for k, label_set in enumerate(r.labels[::-1]):
                if obj_label in label_set:
                    break
            else:
                k += 1

            total_rewind += k

        if isinstance(rewind, int) and rewind > 0:
            total_rewind += rewind

        if total_rewind > 0:
            N = len(r)
            endpt = max(0, N - total_rewind)

            objects = r.objects[:endpt]
            label_sets = r.labels[:endpt]

            r.objects.clear()
            r.labels.clear()
            r.labeled_objects.clear()

            for obj, label_set in zip(objects, label_sets):
                r._add_object(obj, label_set)

            r.N = len(r.objects)

        return r

    def _add_object(self, obj, label_set=None):
        if label_set is None:
            label_set = set()
        self.objects.append(obj)
        self.labels.append(label_set)
        self.N += 1
        if label_set is not None:
            for label in label_set:
                self.labeled_objects[label] = obj

    def label_last_object(self, label):
        """Apply a label after an object has already been added"""
        if len(self) < 1:
            raise BadStepCombination(f'Cannot apply as({label}) to empty result')  # pragma: no cover
        self.labels[-1].add(label)
        self.labeled_objects[label] = self.objects[-1]

    def has_labeled_vertex(self, obj_label):
        return isinstance(self.labeled_objects.get(obj_label), graph.Vertex)

    def has_labeled_edge(self, obj_label):
        return isinstance(self.labeled_objects.get(obj_label), graph.Edge)

    def get_labeled_object(self, obj_label, default=None):
        # In TinkerGraph 3.7.2, it seems a subsequent `as_()` *cannot* override
        # a name that was earlier assigned in a `store()` step. So we check
        # storage lists first.
        if obj_label in self.ss.storage_lists:
            return self.ss.storage_lists[obj_label]
        return self.labeled_objects.get(obj_label, default)

    def get_object_by_index(self, k):
        return self.objects[k]

    def path_is_simple(self):
        """
        Say whether the path is simple, meaning that it never visits any vertex or edge twice.
        """
        # Because vertices and edges share an ID space, we don't need to keep separate
        # ID sets for vertices and edges visited here; we can just use a single set.
        ids = set()
        for obj in self.objects:
            if isinstance(obj, (graph.Vertex, graph.Edge)):
                id_ = obj.id
                if id_ in ids:
                    return False
                ids.add(id_)
        return True

    def note_vertices(self, vertices):
        """Record vertices by ID. """
        for v in vertices:
            self.vertices_by_id[v.id] = v

    def note_edges(self, edges):
        """Record edges by ID. """
        for e in edges:
            self.edges_by_id[e.id] = e

    def form_vertex(self, id, label=None, properties=None):
        """
        Ensure that a vertex exists, forming it if necessary, or potentially
        adding new information to it, such as label or properties. Do *not* add
        the vertex to the path.

        :param id: the vertex id. Required.
        :param label: optional vertex label.
        :param properties: optional list of vertex properties.
        :return: Vertex instance
        """
        # Do we already have this vertex?
        vertex = self.vertices_by_id.get(id)
        if vertex is None:
            # If not, then form a new one.
            vertex = graph.Vertex(id, label=label, properties=properties)
            self.vertices_by_id[id] = vertex
        else:
            # We do already have this vertex. Accept any new info.
            if label is not None:
                vertex.label = label
            if properties is not None:
                vertex.properties = properties
        return vertex

    def add_vertex_to_path(self, id, label=None, properties=None):
        """
        Add a vertex to our path.

        :param id: the vertex id. Required.
        :param label: optional vertex label.
        :param properties: optional list of vertex properties.
        :return: nothing
        """
        vertex = self.form_vertex(id, label=label, properties=properties)
        self._add_object(vertex)

    def form_edge(self, id, outV_id=None, inV_id=None, label=None, properties=None):
        """
        Ensure that an edge exists, forming it if necessary, or potentially
        adding new information to it, such as label or properties. Do *not* add
        the edge to the path.

        :param id: the edge id
        :param outV_id: the id of the "out" or source vertex, if known
        :param inV_id: the id of the "in" or target vertex, if known
        :param label: the edge label, if known
        :param properties: optional list of edge properties.
        :return: Edge instance
        """
        # Do we already have this edge?
        edge = self.edges_by_id.get(id)
        if edge is None:
            # If not, then form a new one.
            if outV_id is None or inV_id is None or label is None:
                outV_id, p, inV_id = qt.run_quads_query(
                    self.con, sel='s, p, o', g=-id, do_close=True, fetch_one=True
                ).fetchone()
                if label is None:
                    label = qt.decode_edge_label(self.con, p)
            outV = self.form_vertex(outV_id)
            inV = self.form_vertex(inV_id)
            edge = graph.Edge(id, outV, label, inV, properties=properties)
            self.edges_by_id[id] = edge
        else:
            # We do already have this edge. Accept any new info.
            if label is not None:
                edge.label = label
            if properties is not None:
                edge.properties = properties
        return edge

    def add_edge_to_path(self, id, outV_id=None, inV_id=None, label=None, properties=None):
        """
        Add an edge to our path.

        :param id: the edge id. Required.
        :param outV_id: the id of the "out" or source vertex, if known
        :param inV_id: the id of the "in" or target vertex, if known
        :param label: the edge label, if known
        :param properties: optional list of edge properties.
        :return: nothing
        """
        edge = self.form_edge(id, outV_id=outV_id, inV_id=inV_id, label=label, properties=properties)
        self._add_object(edge)

    def add_object_to_path(self, signed_id):
        """
        Add a vertex or an edge to the path, depending on sign of ID.

        :param signed_id: positive or negative integer, being the ID of a vertex if positive,
            or the negated ID of an edge if negative.
        :return: nothing
        """
        if signed_id > 0:
            self.add_vertex_to_path(signed_id)
        elif signed_id < 0:
            self.add_edge_to_path(-signed_id)

    def determine_label(self, obj):
        """
        Given any Vertex or Edge, ensure that its label is defined.

        :param obj: Vertex or Edge
        :return: nothing
        """
        if isinstance(obj, (graph.Vertex, graph.Edge)) and obj.label is None:
            obj.label = qt.get_label(self.con, obj.id, self.get_subject_type(obj))

    def determine_properties(self, obj):
        """
        Given any Vertex or Edge, ensure that its properties are defined.

        :param obj: Vertex or Edge
        :return: nothing
        """
        if isinstance(obj, (graph.Vertex, graph.Edge)) and obj.properties is None:
            obj.properties = self.get_properties(obj, as_list=True)

    def hydrate_object(self, obj):
        """
        Given any type of object at all, attempt to hydrate any vertices or edges
        at or within it.

            * If it is a Vertex or Edge then make sure it is completely hydrated,
                i.e. its label and all properties are defined. (It's called "hydration"
                by analogy to organic chemistry, as if we are tacking on all the missing H
                ions to the skeletal structure.)
            * If it is a VertexProperty, hydrate its vertex.
            * If it is a Property, hydrate its element.
            * If it is a path, hydrate its elements.
            * If it is a list, hydrate its elements.
            * If it is a dictionary, hydrate its values.
            * If it is anything else, do nothing.

        :param obj: anything
        :return: nothing
        """
        if isinstance(obj, graph.Path):
            for element in obj.objects:
                self.hydrate_object(element)
        elif isinstance(obj, list):
            for element in obj:
                self.hydrate_object(element)
        elif isinstance(obj, dict):
            for element in obj.values():
                self.hydrate_object(element)
        elif isinstance(obj, graph.VertexProperty):
            self.hydrate_object(obj.vertex)
        elif isinstance(obj, graph.Property):
            self.hydrate_object(obj.element)
        elif isinstance(obj, (graph.Vertex, graph.Edge)):
            self.determine_label(obj)
            self.determine_properties(obj)

    def move_to_value(self, value):
        self._add_object(value)

    def make_path(self, objects):
        """
        Make a graph.Path

        :param objects: list of objects in the path.
        :return: graph.Path
        """
        # Subtle: Need to use a (shallow) copy of `self.labels` rather than a direct reference
        # to it, or else the `Path` object will be modified when it is added as the next object
        # to this `Result`, causing `self.labels` to gain a new entry!
        return graph.Path(self.labels[:], objects)

    def build_element_info_dict(self, obj,
                                incl_tokens=False, incl_endpts=False,
                                prop_names=None, single_props=False):
        """
        For a Vertex or Edge, build a dictionary offering a common superset of the info provided by
        either the `value_map()` or `element_map()` steps.

        :param obj: the Vertex or Edge
        :param incl_tokens: if True, include `id` and `label` entries.
        :param incl_endpts: if True, and `obj` is an `Edge`, include its endpoints as
            element maps with tokens but no properties.
        :param prop_names: if None, provide all properties of the object; otherwise pass a list
            of property names to limit it to just these properties.
        :param single_props: if True, provide only the first value of each property, otherwise
            provide a list of all values. (In both cases, it is property *values*, not objects.)
        :return: dict
        """
        assert isinstance(obj, (graph.Vertex, graph.Edge))

        info_dict = {}
        if incl_tokens:
            self.determine_label(obj)
            info_dict = {T.id: obj.id, T.label: obj.label}

        if incl_endpts and isinstance(obj, graph.Edge):
            info_dict.update({
                Direction.OUT: self.build_element_info_dict(obj.outV, incl_tokens=True, prop_names=[]),
                Direction.IN: self.build_element_info_dict(obj.inV, incl_tokens=True, prop_names=[])
            })

        d = self.get_properties(obj, prop_names=prop_names, single=single_props, value_only=True)
        info_dict.update(d)

        return info_dict

    def set_property(self, obj, prop_name: str, prop_value, cardinality=Cardinality.single):
        """
        Set a property.

        WARNING: In "set" and "single" cardinalities, we do NOT actively "clean things up" for you.
        If you want a set but already have repeats, or you want single but already have multiple, then
        you should first do a `__.properties().drop()` in order to clean things up. Otherwise the results
        will be a mess.

        :param obj: the object (graph.Vertex or graph.Edge) whose property is to be set
        :param prop_name: the name of the property you want to set (str)
        :param prop_value: the value you want to set
        :param cardinality: value of the Cardinality enum
        :return: nothing
        """
        obj_id = obj.id
        is_vertex = isinstance(obj, graph.Vertex)
        is_edge = isinstance(obj, graph.Edge)
        # Our reference implementation TinkerGraph 3.7.2 does not allow multiple props on edges.
        if is_edge and cardinality is not Cardinality.single:
            raise UnsupportedUsage("Cannot set multiple properties on edges")

        if cardinality is Cardinality.list_:
            # In list card., we don't have to look at existing properties. We can just set a new one.
            # Since list card is allowed only for vertices, we know we're working with a vertex.
            qt.create_vertex_property(self.con, obj_id, prop_name, prop_value)
        elif cardinality is Cardinality.set_:
            existing = self.get_properties(obj, prop_names=[prop_name], value_only=True)
            # If this property value has already been set under this property name, we do nothing.
            if prop_name in existing and prop_value in existing[prop_name]:
                return
            # Since set card is allowed only for vertices, we know we're working with a vertex.
            qt.create_vertex_property(self.con, obj_id, prop_name, prop_value)
        else:
            assert cardinality is Cardinality.single
            existing = self.get_properties(obj, prop_names=[prop_name], single=True)
            if prop_name in existing:
                # Prop name has already been set.
                if is_vertex:
                    vp = existing[prop_name]
                    assert isinstance(vp, graph.VertexProperty)
                    if prop_value != vp.value:
                        old_prop_id = vp.id
                        qt.replace_vertex_property(self.con, old_prop_id, prop_value)
                elif is_edge:
                    ep = existing[prop_name]
                    assert isinstance(ep, graph.Property)
                    if prop_value != ep.value:
                        qt.update_edge_property(self.con, obj_id, prop_name, prop_value)
            else:
                # Prop name has not been set yet.
                if is_vertex:
                    qt.create_vertex_property(self.con, obj_id, prop_name, prop_value)
                elif is_edge:
                    qt.record_edge_property(self.con, obj_id, prop_name, prop_value)

    def get_properties(self, obj, prop_names=None, single=False, value_only=False, as_list=False):
        """
        Get properties for a given object

        :param obj: the object (graph.Vertex or graph.Edge) whose properties are sought
        :param prop_names: if None, get all properties. Otherwise, pass a list of strings, and then we
            retrieve only the properties named in this list.
        :param single: set True if you only want the first property under each name, False
            if you want lists of properties for each name.
        :param value_only: set True if you want only the value of each property, False for an instance of
            one of the `VertexProperty` or `Property` graph classes.
        :param as_list: set True if you want a flat list of all properties, False if you want a
            dictionary organizing the properties by name
        :return: dict or list as requested by the `single`, `value_only`, and `as_list` kwargs.
        """
        s = -obj.id if isinstance(obj, graph.Edge) else obj.id
        p = qt.Sign.POS if prop_names is None else qt.write_strings_to_ids_query(prop_names)
        rows = qt.run_quads_query(self.con, sel='p, o, g', s=s, p=p, do_close=True).fetchall()

        d = {} if single else defaultdict(list)
        p_values = set()
        decoded_prop_names = {}

        for p, o, g in rows:
            if p in p_values:
                if single:
                    continue
                property_name = decoded_prop_names[p]
            else:
                p_values.add(p)
                decoded_prop_names[p] = property_name = qt.id_to_string(self.con, p)

            property_value = qt.decode_property_value(self.con, o)

            if value_only:
                prop = property_value
            elif g == qt.Constants.NON_ID_G_VALUE:
                prop = graph.Property(property_name, property_value, obj)
            else:
                assert g > 0
                prop = graph.VertexProperty(g, property_name, property_value, obj)

            if single:
                d[property_name] = prop
            else:
                d[property_name].append(prop)

        if as_list:
            d = sum(d.values(), [])

        return d
