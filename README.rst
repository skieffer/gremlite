.. ...............................................................................
   :   Copyright (c) 2024 Steve Kieffer                                          :
   :                                                                             :
   :   Licensed under the Apache License, Version 2.0 (the "License");           :
   :   you may not use this file except in compliance with the License.          :
   :   You may obtain a copy of the License at                                   :
   :                                                                             :
   :       http://www.apache.org/licenses/LICENSE-2.0                            :
   :                                                                             :
   :   Unless required by applicable law or agreed to in writing, software       :
   :   distributed under the License is distributed on an "AS IS" BASIS,         :
   :   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  :
   :   See the License for the specific language governing permissions and       :
   :   limitations under the License.                                            :
.. ..............................................................................:


Introduction
============

GremLite is a fully functioning, serverless graph database. It uses Python's built-in ``sqlite3``
module to persist your data to disk, and it understands (much of) the Gremlin_ graph query language.
(See language support below.)

Requirements
============

SQLite 3.35 or later is required. You can check your version with:

.. code-block:: shell

   $ python -c "import sqlite3; print(sqlite3.sqlite_version)"

Usage
=====

The ``gremlite`` package is designed to integrate seamlessly with `gremlinpython`_, the official Python package
for connecting to Apache TinkerPop :sup:`TM` graph database systems.

Whereas ordinary usage of ``gremlinpython`` to connect to an acutal Gremlin server might look
like this:

.. code-block:: python

    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.process.anonymous_traversal import traversal

    uri = 'ws://localhost:8182/gremlin'
    remote = DriverRemoteConnection(uri)
    g = traversal().with_remote(remote)

usage with ``gremlite`` instead looks like this:

.. code-block:: python

    from gremlite import SQLiteConnection
    from gremlin_python.process.anonymous_traversal import traversal

    path = '/filesystem/path/to/my_sqlite_database_file.db'
    remote = SQLiteConnection(path)
    g = traversal().with_remote(remote)

That's it. You don't have to set up any tables or indexes. Just start using ``g`` to make
graph traversals as you would with any other graph database.

In the example, we chose to have our on-disk database file live at
``/filesystem/path/to/my_sqlite_database_file.db``. The first time you use a given database file with
GremLite, you must ensure that the directory in which it is to live (like ``/filesystem/path/to`` in our
example) already exists. The file itself however (like ``my_sqlite_database_file.db``) should not yet exist;
GremLite will create it for you. When you want to continue using the same database on subsequent connections,
simply reuse the same path.

Committing your changes
-----------------------

Because all data persistence in ``gremlite`` is through Python's built-in ``sqlite3`` module,
the transaction model is similar. Carrying on with the example above, we could add a vertex and
commit our changes like this:

.. code-block:: python

    g.add_v('cat').property('color', 'black').iterate()
    remote.commit()

Here, we are using the implicit transaction that is started for us when we do not start one
explicitly. All changes pile up in this transaction until we say ``remote.commit()``, and then
they are committed to disk.

Using explicit transactions instead looks like this:

.. code-block:: python

    tx = g.tx()
    g1 = tx.begin()
    g1.add_v('cat').property('color', 'gray').iterate()
    tx.commit()

Or finally, if you prefer, you can instead work in "auto commit" mode:

.. code-block:: python

    remote = SQLiteConnection(path, autocommit=True)
    g = traversal().with_remote(remote)
    g.add_v('cat').property('color', 'orange').iterate()

and all your changes will be immediately commited to disk as
the traversal procedes, without your having to request a commit at any time.
Be aware however that this mode of operation tends to be slower.

Gremlin Language Support
========================

Support for the Gremlin language in ``gremlite`` is not yet 100% complete, but covers a fairly good chunk, with which
you can do a lot. If you are missing an important step, please `open an issue`_.

Currently Supported Steps
-------------------------

The list below is meant only to indicate the set of steps that are supported, and is *not* intended to serve as complete
documentation on the use and meaning of these steps. For that, please see the official `Gremlin documentation`_.

* ``V``

  - 0 args: select all vertices
  - 1 list, tuple, or set of ``Vertex`` instances or ints: select these vertices / the vertices with these IDs
  - 1 or more ``Vertex`` instances or ints: select these vertices / the vertices with these IDs

* ``E``

  - 0 args: select all edges
  - 1 list, tuple, or set of ``Edge`` instances or ints: select these edges / the edges with these IDs
  - 1 or more ``Edge`` instances or ints: select these edges / the edges with these IDs

* ``add_e``

  - 1 string: the edge label

* ``add_v``

  - 0 args: the vertex automatically gets the label "vertex"
  - 1 string: the vertex label

* ``and_``

  - 1 or more traversals: allow the incoming result to pass through iff it produces at
    least one result in *each* of the given traversals.

* ``as_``

  - 1 or more strings: apply these temporary labels to the current object.
  - Inside of a ``where()`` step, instead act as a filter, passing the current object
    iff it is the same as the one already having this label (or these labels).
    See *Practical Gremlin* on `pattern matching using where`_.

* ``barrier``

  - 0 args: First generate *all* results from the foregoing steps, before proceding onward
    with subsequent steps. Like ``fold()``, except that intead of bundling the incoming
    results into a list, they are passed onward one at a time.

* ``both_``

  - 0 args: hop from the current vertex to adjacent vertices along both incoming and outgoing edges
  - 1 or more strings: the edges must have *any* of these labels

* ``both_e``

  - 0 args: move from the current vertex to both its incoming and outgoing edges
  - 1 or more strings: the edges must have *any* of these labels

* ``both_v``

  - 0 args: move from the current edge to both of its endpoint vertices

* ``by`` modifying an ``order``, ``path``, ``project``, ``select``, or ``value_map`` step

  - 0 args: leave object unmodified
  - 1 string: map object to its (first) property value for this property name
  - 1 traversal: map object to first result when following this traversal
  - When modifying an ``order`` step, a final arg may be added, being a value of the
    ``Order`` enum (``asc``, ``desc``, or ``shuffle``). Default ``Order.asc``.
  - When modifying a ``value_map`` step, modification is of the property lists in the map.

* ``cap``

  - 1 string: iterate over all previous steps, and produce the storage list by this name,
    as built by ``store()`` steps

* ``coalesce``

  - 1 or more traversals: carry out the first traversal that returns at least one result

* ``constant``

  - 1 arg: make current object equal to this value

* ``count``

  - 0 args: return the total number of results produced by all the foregoing steps

* ``drop``

  - 0 args: fully drop (delete) the incoming object (property, edge, or vertex) from the database

* ``element_map``

  - 0 args: include all existing properties
  - 1 or more strings: include only properties having these names

* ``emit``

  - 0 args: modify a ``repeat()`` step so it emits all results (may come before or after)

* ``flat_map``

  - 1 traversal: carry out the entire traversal on each incoming result, and produce the
    output as the outgoing result. (Provides a way to group steps together.)

* ``fold``

  - 0 args: gather all incoming results into a single list

* ``has``

  - ``(key)``: keep only those objects that have property ``key`` at all, with no
    constraint on the value.
  - ``(key, value)``: keep only those objects that have property ``key``
    with value ``value``. The ``value`` may be ``None``, boolean, int, float, string,
    or a ``TextP`` or ``P`` operator.
  - ``(label, key, value)``: shorthand for ``.has_label(label).has(key, value)``

* ``has_label``

  - 1 string or ``TextP`` or ``P`` operator: keep only those objects that have a matching label
  - 2 or more strings: keep only those objects that have *any* of these labels

* ``id_``

  - 0 args: return the current object's id

* ``identity``

  - 0 args: return the current object

* ``in_``

  - 0 args: hop from the current vertex to adjacent vertices along incoming edges
  - 1 or more strings: the edges must have *any* of these labels

* ``in_e``

  - 0 args: move from the current vertex to its incoming edges
  - 1 or more strings: the edges must have *any* of these labels

* ``in_v``

  - 0 args: move from the current edge to its target vertex

* ``key``

  - 0 args: map an incoming property to its key

* ``label``

  - 0 args: return the current object's label

* ``limit``

  - 1 int: limit to this many results

* ``none``

  - 0 args: produce no output

* ``not_``

  - 1 traversal: allow the incoming result to pass through iff it does not produce
    any results in the given traversal.

* ``or_``

  - 1 or more traversals: allow the incoming result to pass through iff it produces at
    least one result in *any* of the given traversals.

* ``order``

  - 0 args: like a ``barrier()`` step, except that the incoming results are sorted
    before being emitted.

* ``other_v``

  - 0 args: move from the current edge to that one of its endpoints that was not
    just visited

* ``out_``

  - 0 args: hop from the current vertex to adjacent vertices along outgoing edges
  - 1 or more strings: the edges must have *any* of these labels

* ``out_e``

  - 0 args: move from the current vertex to its outgoing edges
  - 1 or more strings: the edges must have *any* of these labels

* ``out_v``

  - 0 args: move from the current edge to its source vertex

* ``path``

  - 0 args: return the path of objects visited so far

* ``project``

  - 1 or more strings: build a dictionary with these as keys

* ``properties``

  - 0 args: iterate over *all* properties of the incoming object
  - 1 or more strings: restrict to properties having *any* of these names

* ``property``

  - ``(key, value)``: set a property value, with ``single`` cardinality. The ``value`` may be
    ``None``, boolean, int, float, or string.
  - ``(Cardinality, key, value)``: pass a value of the ``gremlin_python.process.traversal.Cardinality`` enum
    to set the property with that cardinality. The ``list_`` and ``set_`` cardinalities are supported
    only on vertices, not on edges.

* ``repeat``

  - 1 traversal: repeat this traversal

* ``select``

  - 1 or more strings: select the objects that were assigned these labels

* ``side_effect``

  - 1 traversal: carry out the traversal as a continuation, but do not return its results; instead,
    return the same incoming results that arrived at this step.

* ``store``

  - 1 string: store the incoming object in a list by this name

* ``times``

  - 1 int: constrain a ``repeat()`` step to apply its traversal at most this many times.

* ``unfold``

  - 0 args: iterate over an incoming list as separate results

* ``union``

  - 0 or more traversals: produce all the results produced by these traversals, in the order given.
    (Repeats are *not* eliminated.)

* ``until``

  - 1 traversal: modify a ``repeat()`` step so it emits but does not go beyond results that satisfy the
    given traversal. May come before or after the ``repeat()`` step.

* ``value``

  - 0 args: map an incoming property to its value

* ``value_map``

  - 0 args: include all existing properties
  - 1 or more strings: include only properties having these names
  - a boolean arg may be prepended to any of the above cases, to say whether the
    ID and label of the object should be included (default ``False``)

* ``values``

  - ``values(*args)`` is essentially a shorthand for ``properties(*args).value()``
  - 0 args: iterate over *all* properties of the incoming object, and produce only the value,
    not the whole property.
  - 1 or more strings: restrict to properties having *any* of these names

* ``where``

  - 1 traversal: allow the incoming result to pass through iff it produces at
    least one result in the given traversal.
    Note: This may seem like an ``and()`` step restricted to a single traversal, but it is
    actually more powerful because it can also do pattern matching; see ``as_()`` step.


Support for Predicates
----------------------

At this time, Gremlin's ``P`` and ``TextP`` predicates are supported only in the ``value``
arguments to the ``has()`` and ``has_label()`` steps, and only for the operators listed below.

Support should be easy to extend to other steps and other operators; we just haven't bothered
to do it yet. So if you are missing something, please `open an issue`_.

* ``TextP``

  - ``starting_with``
  - ``containing``
  - ``ending_with``
  - ``gt``
  - ``lt``
  - ``gte``
  - ``lte``

* ``P``

  - ``within``


.. _Gremlin: https://tinkerpop.apache.org/gremlin.html
.. _gremlinpython: https://pypi.org/project/gremlinpython/
.. _open an issue: https://github.com/skieffer/gremlite/issues
.. _Gremlin documentation: https://tinkerpop.apache.org/docs/current/reference/#graph-traversal-steps
.. _pattern matching using where: https://kelvinlawrence.net/book/Gremlin-Graph-Guide.html#patternwhere
