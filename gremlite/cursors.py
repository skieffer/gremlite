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

from abc import ABC


class StandinCursor(ABC):
    """
    Common base class for any of our "phony cursor" objects that can stand in for
    sqlite cursors in our Producer iteration process.
    """

    def fetchone(self):
        raise NotImplementedError  # pragma: no cover


class DummyCursor(StandinCursor):
    """
    Not a true cursor, in that it's not representing a generator or stream, but
    instead simply wraps a list of already generated rows.
    """

    def __init__(self, rows):
        """
        :param rows: list of objects the cursor should return
        """
        self.rows = rows
        self.N = len(self.rows)
        self.ptr = 0

    @staticmethod
    def emtpy_cursor():
        """Make a dummy cursor that has no rows"""
        return DummyCursor([])

    @staticmethod
    def one_time_cursor():
        """Make a dummy cursor that has one row"""
        return DummyCursor([[]])

    def fetchone(self):
        if self.ptr < self.N:
            row = self.rows[self.ptr]
            self.ptr += 1
            return row
        # Mimic the behavior of sqlite cursors. When they are exhausted, they return `None`.
        return None

    def fetchall(self):
        rows = self.rows[self.ptr:]
        self.ptr = self.N
        return rows


class GeneratorCursor(StandinCursor):
    """
    Accepts a generator and operates it as a cursor.
    """

    def __init__(self, gen):
        self.gen = gen

    def fetchone(self):
        try:
            item = next(self.gen)
        except StopIteration:
            return None
        return item
