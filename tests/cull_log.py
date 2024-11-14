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

"""
For culling results of interest out of pytest.log
"""

import re
import sys


def cull_log(log_file_path='pytest.log'):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    indices_used = set()
    quad_query_constraints_used = set()

    for line in lines:
        M = re.search(r'USING (COVERING )?INDEX (\w+)', line)
        if M:
            indices_used.add(M.group(2))
        M = re.search(r'QUAD QUERY CONSTRAINTS: (\[\w*\|\w*])', line)
        if M:
            quad_query_constraints_used.add(M.group(1))

    return indices_used, quad_query_constraints_used


def main(log_file_path='pytest.log'):
    indices_used, quad_query_constraints_used = cull_log(log_file_path)

    print()
    print("Indices used:")
    print('  ', indices_used)

    print()
    print('Quad query formats:')
    for q in sorted(quad_query_constraints_used, reverse=True):
        print('  ', q)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'pytest.log'

    main(path)
