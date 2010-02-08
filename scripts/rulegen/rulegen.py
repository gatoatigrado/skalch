#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2010 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, pprint, process_jinja2, set, sort_asc, sort_desc)
import pygtk; pygtk.require("2.0")
import gtk

modpath = Path(__file__).parent()
proj_path = modpath.parent().parent()
state_path = Path("~/.config/rulegen.pickle")

class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)

class State(object):
    def __init__(self):
        self.filepath = proj_path.subpath("base/src/test/scala/angelic/simple/SugaredTest.scala")

    @classmethod
    def load(cls):
        default = cls()
        if state_path.isfile():
            state = state_path.unpickle()
            if isinstance(state, cls):
                state.__dict__.update(dict(k, v) for k, v in default.__dict__.items() if not k in state.__dict__)
                return state
        return default

class GuiBase(object):
    def __init__(self, state):
        self.state = state

        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("Skalch compiler rulegen (developer tool)")
        self.window.connect("delete_event", self.delete_event)
        self.window.connect("destroy", self.destroy)
        self.initialize_widgets()

    def initialize_widgets(self):
        window = self.window
        widgets = [window]
        def w(v): widgets.append(v); return v


        # menubar
        menubar = w(self.get_menubar())
        menubox = w(gtk.VBox(False, 0))
        menubox.pack_start(menubar, False, False, 0)
        window.add(menubox)

        main_box = w(gtk.HBox(False, 0))
        src_view = w(gtk.TextView())
        buf = src_view.get_buffer()
        src_view.set_editable(False)
        buf.set_text(self.state.filepath.read())
        main_box.add(src_view)
        menubox.add(main_box)


        # for the right box...
        right_box = w(gtk.VBox(False, 0))
        graphlet = w(gtk.Button("graphlet"))
        graphlet.connect("clicked", self.get_graphlet)
        right_box.pack_start(graphlet, False, False, 0)

        gxltosketch = w(gtk.Button("gxltosketch"))
        gxltosketch.connect("clicked", self.get_gxltosketch)
        right_box.pack_start(gxltosketch, False, False, 0)
        main_box.add(right_box)


        vars = globals(); vars.update(locals())
        def objname(t): return get_singleton([k for k, v in vars.items() if v == t])
        self.widgets = ObjFromDict(dict((objname(v), v) for v in widgets))
        assert self.widgets.main_box == main_box

        self.window.show_all()

    def get_menubar(self):
        # adapted from http://zetcode.com/tutorials/pygtktutorial/menus/
        mb = gtk.MenuBar()

        filem = gtk.MenuItem("_File")
        filemenu = gtk.Menu()
        filem.set_submenu(filemenu)

        agr = gtk.AccelGroup()
        self.window.add_accel_group(agr)

        openm = gtk.ImageMenuItem(gtk.STOCK_OPEN, agr)
        key, mod = gtk.accelerator_parse("O")
        openm.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        filemenu.append(openm)

        sep = gtk.SeparatorMenuItem()
        filemenu.append(sep)

        exit = gtk.ImageMenuItem(gtk.STOCK_QUIT, agr)
        key, mod = gtk.accelerator_parse("Q")
        exit.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)

        exit.connect("activate", gtk.main_quit)
        filemenu.append(exit)
        mb.append(filem)

        genm = gtk.MenuItem("_Generate")
        generatemenu = gtk.Menu()
        genm.set_submenu(generatemenu)

        graphlet = gtk.MenuItem("graph_let")
        graphlet.connect("activate", self.get_graphlet)
        key, mod = gtk.accelerator_parse("L")
        graphlet.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        generatemenu.append(graphlet)

        gxltosketch = gtk.MenuItem("gxltos_ketch")
        gxltosketch.connect("activate", self.get_gxltosketch)
        key, mod = gtk.accelerator_parse("K")
        gxltosketch.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        generatemenu.append(gxltosketch)

        mb.append(genm)

        return mb

    def get_graphlet(self, widget):
        print("get graphlet...")

    def get_gxltosketch(self, widget):
        print("get gxltosketch...")

    def delete_event(self, widget, event, data=None):
        return False

        # Another callback
    def destroy(self, widget, data=None):
        gtk.main_quit()

    def main(self):
        gtk.main()

def main():
    state = State.load()
    base = GuiBase(state)
    base.main()
    state_path.pickle(state)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser()
    # cmdopts.add_option("--myflag", action="store_true", help="set my flag variable")
    noptions = len(cmdopts.option_list) - 1
    varargs = bool(main.__code__.co_flags & 0x04)
    required_args = main.__code__.co_argcount - noptions
    if varargs:
        cmdopts.usage = "%%prog [options] <<list %s>>" % (main.__code__.co_varnames[0])
    else:
        cmdopts.usage = "%prog [options] " + " ".join(
            v for v in main.__code__.co_varnames[:required_args])
    options, args = cmdopts.parse_args()
    if not varargs and required_args != len(args):
        cmdopts.error("%d arguments required." % (required_args))
    main(*args, **options.__dict__)
