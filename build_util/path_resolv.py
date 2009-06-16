#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE - this file is mostly for examining compilation results, not for building the project

from __future__ import print_function
from collections import namedtuple
import optparse, os, shutil, subprocess, types

home=os.environ["HOME"]
assert os.path.isdir(home), "home directory must be present"

class Path(str):
    def __new__(cls, *unix_paths):
        args = []
        for path in unix_paths:
            v = path[1:].split("/") # abs path handling
            v[0] = path[0] + v[0]
            args.extend(v)
        args[0] = home if args[0] == "~" else args[0]
        path = os.path.realpath(os.path.join(*args))
        return str.__new__(cls, path)

    def __getattr__(self, key):
        if key in dir(os):
            fcn = getattr(os, key)
        elif key in dir(shutil):
            fcn = getattr(shutil, key)
        else:
            assert key in dir(os.path), key
            fcn = getattr(os.path, key)
        assert (type(fcn) == types.FunctionType or
            type(fcn) == types.BuiltinFunctionType)
        def inner(*argv, **kwargs):
            result = fcn(self, *argv, **kwargs)
            if type(result) == str:
                return Path(result)
            else:
                return result
        return inner

    def subpath(self, *argv):
        return Path(self, *argv)

    def remove_tree(self):
        self.rmtree() if self.isdir() else self.unlink()

    def read(self):
        return open(self, "r").read()

    def __repr__(self): return "Path[\"%s\"]" %(self)

    def ifexists(self, debug=False):
        if debug and not self.exists():
            print("    debug -- path '%s' doesn't exist." %(debug))
            assert False
        return self if self.exists() else None

    @classmethod
    def pathjoin(cls, *argv):
        return os.path.pathsep.join([str(v) for v in argv])

class PathResolver:
    def get_file(self, basename):
        raise NotImplementedError, "abstract"

class EnvironPathResolver:
    def __init__(self):
        self.dir_arr = []
        for k, v in os.environ.items():
            if "HOME" in k or k == "CLASSPATH":
                self.dir_arr.extend([v1 for v1 in v.split(os.pathsep) if v1])
    def get_file(self, basename):
        for dir_ in self.dir_arr:
            if Path(dir_, basename).exists():
                return Path(dir_, basename)

class DefaultPathResolver:
    def get_file(self, basename):
        # look in ~/sandbox/eclipse
        return Path("~/sandbox/eclipse", basename).ifexists(debug=True)

resolvers = [EnvironPathResolver(), DefaultPathResolver()]

def resolve(path):
    for resolver in resolvers:
        result = resolver.get_file(path)
        if result:
            return result

class ExecuteIn:
    def __init__(self, path):
        self.prev_path = Path(".")
        self.path = path

    def __enter__(self):
        self.path.chdir()

    def __exit__(self, type_, value, tb):
        self.prev_path.chdir()
