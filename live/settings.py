from __future__ import absolute_import

import importlib
import os
import sys

import live.settings_base as base_settings


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def import_path(fullpath):
    """
    Import a file with full path specification. Allows one to
    import from anywhere, something __import__ does not do.
    """
    path, filename = os.path.split(fullpath)
    filename, ext = os.path.splitext(filename)
    sys.path.insert(0, path)
    print(filename, path)
    module = importlib.import_module(filename, path)
    importlib.reload(module)  # Might be out of date
    del sys.path[0]
    return module


user_settings = import_path(os.path.join('live', 'settings'))
symbol_settings = None
symbol = sys.argv[1] if len(sys.argv) > 1 else None
if symbol:
    print("Importing symbol settings for %s..." % symbol)
    try:
        symbol_settings = import_path(os.path.join('..', 'settings-%s' % symbol))
    except Exception as e:
        print("Unable to find settings-%s.py." % symbol)

# Assemble settings.
settings = {}
settings.update(vars(base_settings))
settings.update(vars(user_settings))
if symbol_settings:
    settings.update(vars(symbol_settings))

# Main export
settings = DotDict(settings)
