"""Run a ppt-master script without reading the restricted Windows MIME registry."""

import mimetypes
import os
import runpy
import sys


mimetypes.MimeTypes.read_windows_registry = lambda self: None
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
