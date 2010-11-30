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

import pygtk; pygtk.require("2.0")
import gobject, gtk

class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)

def async(fcn):
    gobject.idle_add(fcn)

def initGtk():
    gtk.gdk.threads_init()
    gtk.gdk.threads_enter()

class GuiWindow(object):
    def __init__(self, events):
        self.events = events
        events.ui = self
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("Skalch2 GUI tool")
        self.window.connect("delete_event", events.exit)
        self.window.connect("destroy", events.exit)
        self.initialize_keyboard_shortcuts()
        self.initialize_widgets()
        gtk.gdk.threads_leave()
        events.oninit()

    def gtk_err(self, text):
        dialog = gtk.MessageDialog(self.window,
            type=gtk.MESSAGE_ERROR,
            message_format=text, buttons=gtk.BUTTONS_OK)
        dialog.run()
        dialog.destroy()

    def exit(self, *argv, **kwargs):
        self.events.exit()

    def initialize_keyboard_shortcuts(self):
        self.accel_group = gtk.AccelGroup()
        self.window.add_accel_group(self.accel_group)
        #--------------------------------------------------
        # accel_group.connect_group(ord('q'), gtk.gdk.CONTROL_MASK, gtk.ACCEL_LOCKED, gtk.main_quit)
        #-------------------------------------------------- 

        key, mod = gtk.accelerator_parse("Escape")
        self.accel_group.connect_group(key, mod, gtk.ACCEL_LOCKED, self.events.abort_threads)

    def initialize_widgets(self):
        window = self.window
        widgets = [window]
        def w(v):
            widgets.append(v)
            return v
        def hbox(): return w(gtk.HBox(False, 0))
        def vbox(): return w(gtk.VBox(False, 0))
        def button(label, func, shortcut=None):
            result = w(gtk.Button(label))
            result.connect("clicked", func)
            if shortcut:
                key, mod = gtk.accelerator_parse(shortcut)
                result.add_accelerator("activate", self.accel_group, key, mod, gtk.ACCEL_VISIBLE)
            return result



        # menubar, top level box
        menubar = w(self.get_menubar())
        menubox = w(gtk.VBox(False, 0))
        menubox.pack_start(menubar, False, False, 0)
        window.add(menubox)

        # all other containers
        main_box = w(gtk.VBox(False, 0))
        # we don't need the frame object after adding it
        def framebox(text, inner):
            frame = w(gtk.Frame(text))
            frame.add(inner)
            main_box.add(frame)
            return inner
        build_box = framebox("Build", vbox())
        settings_box = framebox("Settings", hbox())
        compiler_box = framebox("Compile", hbox())
        results_box = framebox("View intermediate", vbox())
        menubox.add(main_box)

        #--------------------------------------------------
        # src_view = w(gtk.TextView())
        # buf = src_view.get_buffer()
        # src_view.set_editable(False)
        # src_view.connect("button-release-event",
        #     self.get_selection_info)
        # buf.set_text(self.state.filepath.read())
        # main_box.add(src_view)
        #-------------------------------------------------- 



        # for the right box...
        #--------------------------------------------------
        # right_box = w(gtk.VBox(False, 0))
        # info_label = w(gtk.Label("select some text..."))
        # right_box.pack_start(info_label, False, False, 0)
        # gxltosketch_view = w(gtk.TextView())
        # main_box.add(right_box)
        #-------------------------------------------------- 


        # Build Box
        build_buttons_box = w(gtk.HBox(False, 0))
        build_buttons_box.add(button("Build (Ctrl+B)", self.events.build, "<Control>b"))
        build_buttons_box.add(button("Rebuild", self.events.rebuild))
        status = w(gtk.Label("<b>Status: </b>"))
        status.set_use_markup(True)
        build_box.add(build_buttons_box)
        build_box.add(status)



        # for the tabbed output boxes
        #--------------------------------------------------
        # notebook = w(gtk.Notebook())
        # sk_preproc = w(gtk.
        #-------------------------------------------------- 



        vars = globals(); vars.update(locals())
        def objname(t): return get_singleton([k for k, v in vars.items() if v == t])
        self.widgets = ObjFromDict(dict((objname(v), v) for v in widgets if v in vars.values()))
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
        key, mod = gtk.accelerator_parse("<Control>Q")
        exit.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)

        exit.connect("activate", self.exit)
        filemenu.append(exit)
        mb.append(filem)

        #--------------------------------------------------
        # genm = gtk.MenuItem("_Generate")
        # generatemenu = gtk.Menu()
        # genm.set_submenu(generatemenu)
        #-------------------------------------------------- 

        #--------------------------------------------------
        # graphlet = gtk.MenuItem("graph_let")
        # graphlet.connect("activate", self.get_graphlet)
        # key, mod = gtk.accelerator_parse("L")
        # graphlet.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        # generatemenu.append(graphlet)
        #-------------------------------------------------- 

        #--------------------------------------------------
        # gxltosketch = gtk.MenuItem("gxltos_ketch")
        # gxltosketch.connect("activate", self.get_gxltosketch)
        # key, mod = gtk.accelerator_parse("K")
        # gxltosketch.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        # generatemenu.append(gxltosketch)
        #-------------------------------------------------- 

        #--------------------------------------------------
        # mb.append(genm)
        #-------------------------------------------------- 
        return mb

    def updateStatus(self, message):
        async(lambda: self.widgets.status.set_label("<b>Status: %s</b>" %(message)))
