"""
Utility functions for yt.frontends.einsteintoolkit

"""

import re

def lazy_property(func):
    attr_name = f'_{func.__name__}'

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property

def determine_refinement_factor(parameters, default=2):
    try:
        return int(re.search('Carpet::refinement_factor[\s*]=[\s*](\d+)', parameters, re.IGNORECASE).groups()[0])
    except:
        return default
