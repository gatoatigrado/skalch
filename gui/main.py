#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2010 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple, defaultdict
import sys; reload(sys); sys.setdefaultencoding('utf-8') # print unicode
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, persistent_var, pprint, process_jinja2, set, sort_asc,
    sort_desc)

import ui

import pygtk; pygtk.require("2.0")
import gtk
import time
import types
from eventthread import EventThread

def nodec(fcn):
    fcn.nodec = True
    return fcn

def main(filename, recompile):
    filename = Path(filename)
    assert filename.endswith("scala"), "expecting .scala file"
    scala_gxl = Path(filename.splitext()[0] + ".scala.ast.gxl")
    sketch_gxl = Path(filename.splitext()[0] + ".sketch.ast.gxl")
    assert filename.isfile(), "must be given a filename"

    # NOTE -- EVERYTHING here becomes async magically
    class Events:
        def __init__(self):
            for attr in dir(self):
                value = getattr(self, attr)
                if type(value) == types.MethodType and not getattr(value, "nodec", False):
                    setattr(self, attr, EventThread.decorate(value))

        def oninit(self):
            if recompile:
                if not self.compile_compiler():
                    return
            self.build()

        def build(self, widget=None, event=None):
            self.ui.updateStatus("Building")

            # delete old stuff
            if scala_gxl.isfile():
                scala_gxl.unlink()
            if sketch_gxl.isfile():
                sketch_gxl.unlink()

            # call maven to compile
            self.ui.updateStatus("Maven compile")
            try:
                SubProc(["mvn", "compile", "test-compile"]).start_wait()
            except SubProc.Error:
                self.ui.updateStatus("Maven compile failed (see command line)")
                return

            self.ui.updateStatus("Done")
            return True

        # for now, same thing as build, but when jpype maven & scala compiler integration
        # is completed, it will clear caches and such
        def rebuild(self, widget=None, event=None):
            self.build()

        @nodec
        def abort_threads(self, *argv, **kwargs):
            print("NOTE -- aborting threads upon user request", file=sys.stderr)
            EventThread.stop_all()

        @nodec
        def exit(self, *argv, **kwargs):
            EventThread.stop_all()
            gtk.main_quit()

        def compile_compiler(self, widget=None, event=None):
            self.ui.updateStatus("Compiling Skalch")
            try:
                SubProc(["make", "compile"]).start_wait()
            except SubProc.Error:
                self.ui.updateStatus("Compiling Skalch failed (see command line)")
                return

            self.ui.updateStatus("Compiling Skalch and dependencies done")
            return True

    ui.initGtk()
    window = ui.GuiWindow(Events())
    gtk.main()

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] filename",
        description="description")
    cmdopts.add_option("--recompile", action="store_true", help="compile before startup build")
    options, args = cmdopts.parse_args()
    if len(args) < 0:
        cmdopts.error("invalid number of arguments")
    try:
        main(*args, **options.__dict__)
    except TypeError, e:
        if "main()" in str(e):
            cmdopts.error("insufficent number of arguments")
        else: raise
