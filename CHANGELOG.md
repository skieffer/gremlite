## v0.37.0

- When an `SQLiteConnection` instance is formed, check the version of SQLite that is
  being used, and raise an exception if not version 3.35 or later. This is needed at
  least for our use of the `RETURNING` keyword.

- Add the package requirement that `gremlinpython` be less than `3.8.0`.
  At this time, the only known divergence issue is in the change of `none()`
  to `discard()`. See [the upgrade guide](https://tinkerpop.apache.org/docs/3.8.0/upgrade/#none-and-discard).
  There may well be others; no effort has been made to check at this time.  

  Accordingly, we now advance the `gremlite` version number to `0.37.0`, with the
  intention that our minor version number will indicate which TinkerPop release we
  intend to work with.
