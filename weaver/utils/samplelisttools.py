##################################
# Tools for reading sample lists #
##################################

# The sample list is supposed to be a .yaml file of the following format:
"""
- <file pattern 1>
- <file pattern 2>
- <etc>
"""
# or of the following format:
"""
<name 1>:
    - <file pattern 1a>
    - <file pattern 1b>
<name 2>:
    - <etc>
"""

# This format is then trivially parsed into the same format
# used for directly specifying the input files on the command line,
# see the main README for more info.


import os
import sys
import yaml


def read_sample_list(samplelist):

    # read the yaml file
    with open(samplelist) as stream:
        try: contents = yaml.safe_load(stream)
        except yaml.YAMLError as exc: raise Exception(exc)

    # do parsing
    patterns = []
    if isinstance(contents, list):
        patterns = contents
    elif isinstance(contents, dict):
        for key, val in contents.items():
            for el in val:
                pattern = f'{key}:{el}'
                patterns.append(pattern)

    # return the file patterns
    return patterns
