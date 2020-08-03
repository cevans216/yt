"""
Utility functions for yt.frontends.einsteintoolkit

"""

def lazy_property(func):
    attr_name = f'_{func.__name__}'

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property
