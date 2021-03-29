from __future__ import absolute_import
import os.path 
import sys

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir,'..','model')

if lib_path not in sys.path:
    sys.path.insert(0,lib_path)