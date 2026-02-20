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

"""Exception classes"""


class GremliteError(Exception):
    ...


class BadStepArgs(GremliteError):
    """Bad arguments to Gremlin step"""
    ...


class BadStepCombination(GremliteError):
    """A Gremlin step follows the wrong kind of previous step"""
    ...


class BadDatabase(GremliteError):
    """Database is not as we need."""
    ...


class UnknownStep(GremliteError):
    """Step is not (yet) supported"""
    ...


class UnsupportedUsage(GremliteError):
    """Usage is not supported"""
    ...


class UnexpectedQuadConstraintPattern(GremliteError):
    """A quads table query was made with unexpected constraint pattern"""
    ...
