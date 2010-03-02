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
import re, sys
from multiprocessing import Process
from amara import bindery
import resource

mbyte = 1 << 20
resource.setrlimit(resource.RLIMIT_AS, (1600 * mbyte, 1600 * mbyte))

modpath = Path(__file__).parent()
proj_path = modpath.parent().parent()
state_path = Path("~/.config/rulegen.pickle")

sys.path.append(proj_path.subpath("plugin/src/main/grgen"))
from transform_sketch import main as transform_gxl_main
from transform_sketch import GrgenException

class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)

class State(object):
    def __init__(self):
        self.filepath = proj_path.subpath("base/src/test/scala/angelic/simple/SugaredTest.scala")

    def onload(self):
        self.gxlpath = Path(self.filepath + ".ast.gxl")

    def check_bounds(self, left, right):
        lines = open(self.filepath).readlines()
        selection = "".join([c for line_idx, line in enumerate(lines) for char_idx, c in enumerate(line)
            if left <= (line_idx, char_idx) < right])
        print("you selected", selection)
        return True

    def save(self):
        state_path.pickle(self)

    @classmethod
    def load(cls):
        default = cls()
        default.onload()
        if state_path.isfile():
            try:
                state = state_path.unpickle()
                if isinstance(state, cls):
                    state.__dict__.update(dict((k, v) for k, v in default.__dict__.items() if not k in state.__dict__))
                    state.onload()
                    return state
            except: pass
        return default

class VarnameGen(object):
    CC_RE = re.compile("([a-z])([A-Z])")
    TYPMAP = {
        "FcnArgList": "args",
        "ThisVarRef": "ths",
        "IntConstant": "int_const",
        "If": "if_"
        }
    def __init__(self):
        self.names = set()

    def new_varname(self, typ):
        # TODO -- typical names
        if typ in self.TYPMAP:
            typname = self.TYPMAP[typ]
        else:
            def replace(m): return m.group(1) + "_" + m.group(2)
            typname = self.CC_RE.sub(replace, typ).lower()
        names = (typname + (str(v) if v else "") for v in range(100))
        next_name = (v for v in names if not v in self.names).next()
        self.names.add(next_name)
        return next_name

class NewNodeToken(namedtuple("NewNodeToken", "name type")):
    def __str__(self): return "%s:%s" % (self.name, self.type)

class ExistingNodeToken(namedtuple("ExistingNodeToken", "name")):
    def __str__(self): return self.name

class NewEdgeToken(namedtuple("NewEdgeToken", "name type")):
    def __str__(self): return "-%s:%s->" % (self.name, self.type)

class SyntacticToken(object):
    def __init__(self, symbol): self.symbol = symbol
    def __str__(self): return self.symbol
    def __repr__(self): return "S[%r]" % (self.symbol)

EndChainToken = SyntacticToken(";\n")
ListStartToken = SyntacticToken(' {{ macros.finiteList([')
ListEndToken = SyntacticToken(']) }}')
ListEltQuote = SyntacticToken('"')

def edge_property_iterator(arr, *fcns):
    for i, elt in enumerate(arr):
        left_edges, right_edges = [], []
        if i > 0:
            left_edges = [v for v in fcns if v(arr[i - 1], elt)]
        if i < len(arr) - 1:
            right_edges = [v for v in fcns if v(elt, arr[i + 1])]
        yield left_edges, elt, right_edges

def elementChildren(node):
    return (v for v in node.xml_children if v.xml_type == "element")

class MatchStringBuilder(object):
    def __init__(self, varnamegen=None):
        self.varnamegen = varnamegen if varnamegen is not None else VarnameGen()
        self.nodeNameMap = { }
        self.edgeNameMap = { }
        self.matchString = []
        self.chainLast = None

    def __str__(self):
        indent = ""
        result = ""

        # "properties" of sequential elements
        istoken = lambda a: isinstance(a, (NewNodeToken, ExistingNodeToken, NewEdgeToken))
        def true_for_both(fcn):
            return lambda a, b: fcn(a) and fcn(b)
        tokentoken = true_for_both(istoken)
        endend = true_for_both(lambda a: a == EndChainToken)
        quotequote = true_for_both(lambda a: a == ListEltQuote)

        for left_edges, elt, right_edges in edge_property_iterator(self.matchString,
                tokentoken, endend, quotequote):

            if tokentoken in left_edges:
                result += " "
            if endend in left_edges:
                continue
            elif quotequote in left_edges:
                result += ", "
            result += str(elt)
        return result

    def addNodeToken(self, node):
        self.chainLast = node
        if node in self.nodeNameMap:
            self.matchString.append(ExistingNodeToken(self.nodeNameMap[node]))
        else:
            varname = self.varnamegen.new_varname(node.type)
            self.nodeNameMap[node] = varname
            self.matchString.append(NewNodeToken(varname, node.type))

    def addEdgeToken(self, edge):
        assert edge not in self.edgeNameMap, "edges shouldn't be used twice."
        varname = self.varnamegen.new_varname(edge.edgetype)
        self.edgeNameMap[edge] = varname
        self.matchString.append(NewEdgeToken(varname, edge.edgetype))

    def addNode(self, node):
        assert node.xml_name[-1] == u"node" # must be a node type

        self.addNodeToken(node)
        for child in elementChildren(node):
            if self.chainLast != node:
                self.matchString.append(EndChainToken)
                self.addNodeToken(node)
            self.addEdgeToken(child)
            if child.xml_name[-1] == u"subtree":
                self.addNode(child.node)
            elif child.xml_name[-1] == u"list":
                self.addList(child)
            else:
                assert False, "unknown node type %s" % (child.xml_name[-1])
        self.matchString.append(EndChainToken)

    def addList(self, lst):
        assert lst.xml_name[-1] == u"list"
        lst.type = "List"
        self.matchString.append(ListStartToken)
        for child in elementChildren(lst):
            self.matchString.append(ListEltQuote)
            self.addNodeToken(child.node)
            self.matchString.append(ListEltQuote)
        self.matchString.extend([ListEndToken, EndChainToken])
        for child in elementChildren(lst):
            children = list(elementChildren(child.node))
            if any(children):
                self.addNode(child.node)

#def tmpdebug():
#    xmlText = modpath.subpath("tmp.xml").read()
#    print(xmlText)
#    xml = bindery.parse(xmlText).infos
#    toplevel_elts = elementChildren(xml)
#
#    matchString = MatchStringBuilder()
#    [matchString.addNode(v.node) for v in toplevel_elts]
#    print(str(matchString))
#
#tmpdebug()
#exit()

class GuiBase(object):
    def __init__(self, state):
        self.state = state

        self.subprocs = []

        gtk.gdk.threads_init()
        gtk.gdk.threads_enter()
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("Skalch compiler rulegen (developer tool)")
        self.window.connect("delete_event", self.delete_event)
        self.window.connect("destroy", self.destroy)
        self.initialize_widgets()
        gtk.gdk.threads_leave()

    def assertwrapper(fcn):
        def assertwrapper_inner(self, *argv, **kwargs):
            try:
                return fcn(self, *argv, **kwargs)
            except AssertionError, e:
                self.gtk_err(str(e))
            except GrgenException, e:
                self.gtk_err("=== GrGen Exception ===\n%s" % (e))
        return assertwrapper_inner

    def gtk_err(self, text):
        dialog = gtk.MessageDialog(self.window,
            type=gtk.MESSAGE_ERROR,
            message_format=text, buttons=gtk.BUTTONS_OK)
        dialog.run()
        dialog.destroy()

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
#        import ipdb; ipdb.set_trace()
        src_view.connect("button-release-event", self.get_selection_info)
        buf.set_text(self.state.filepath.read())
        main_box.add(src_view)
        menubox.add(main_box)


        # for the right box...
        right_box = w(gtk.VBox(False, 0))
        info_label = w(gtk.Label("select some text..."))
        right_box.pack_start(info_label, False, False, 0)
        gxltosketch_view = w(gtk.TextView())
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

        ycomp = gtk.MenuItem("display selection in _ycomp")
        ycomp.connect("activate", self.get_ycomp)
        key, mod = gtk.accelerator_parse("Y")
        ycomp.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
        generatemenu.append(ycomp)

        mb.append(genm)

        return mb

    @assertwrapper
    def get_selection_info(self, widget, event):
        bounds = self.try_get_bounds()
        if bounds:
            left, right = bounds
#            self.widgets.info_label.set_text("getting selection info")
            toplevel_elts = []
            infos = self.run_gxl_inner(get_sel_info=True)
            xmlText = "<infos>" + "".join(v.lstrip("[INFO] ").strip() for v in infos) + "</infos>"
            xml = bindery.parse(xmlText).infos
            toplevel_elts = [v for v in xml.xml_children]

            self.widgets.info_label.set_text("selection %r - %r\ntoplevel elt(s):\n%s" % (
                left, right, ", ".join(v.node.type for v in toplevel_elts)))

            matchString = MatchStringBuilder()
            [matchString.addNode(v.node) for v in toplevel_elts]

            print("\n\n\n%s" % (matchString))

            #import ipdb; ipdb.set_trace()

    @assertwrapper
    def get_graphlet(self, widget):
        self.run_gxl_inner(get_graphlet=True)

    @assertwrapper
    def get_ycomp(self, widget):
        # NOTE -- ycomp cannot be synchronous at the time...
        proc = Process(target=self.run_gxl_inner,
            kwargs={"ycomp_selection": True, "ycomp": True})
        proc.start()
        self.subprocs.append(proc)

    def run_gxl_inner(self, **kwargs):
        with ExecuteIn(proj_path):
            SubProc(["make", "gen"]).start_wait()

        bounds = self.try_get_bounds()
        assert bounds, "select some text first."
        left, right = bounds

        assert self.state.gxlpath.isfile(), "run the scala compiler [with the plugin]\n" \
            "on the source first to generate the GXL file."
        return transform_gxl_main(gxl_file=self.state.gxlpath,
            left_sel=left, right_sel=right, silent=True, **kwargs)

    def try_get_bounds(self):
        buf = self.widgets.src_view.get_buffer()
        bounds = buf.get_selection_bounds()
        if len(bounds) != 2:
            return None
        return tuple((v.get_line() + 1, v.get_line_offset()) for v in bounds)

    def get_gxltosketch(self, widget):
        print("get gxltosketch...")

    def delete_event(self, widget, event, data=None):
        return False

        # Another callback
    def destroy(self, widget, data=None):
        gtk.main_quit()
        [v.terminate() for v in self.subprocs]
        SubProc(["killall", "mono"]).start()

    def main(self):
        gtk.main()

def main():
    state = State.load()
    base = GuiBase(state)
    base.main()
    state.save()

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
