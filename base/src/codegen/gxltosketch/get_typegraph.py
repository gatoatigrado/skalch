#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

from __future__ import division, print_function
from collections import namedtuple
from warnings import warn
import xml.dom.minidom as xml

try:
    from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
        list, memoize_file, pprint, set, sort)
except:
    raise ImportError("please install gatoatigrado's utility library from "
            "bitbucket.org/gatoatigrado/gatoatigrado_lib")

GXL_NODE_TYPE = "http://www.gupro.de/GXL/gxl-1.0.gxl#NodeClass"
GXL_EDGE_TYPE = "http://www.gupro.de/GXL/gxl-1.0.gxl#EdgeClass"
GXL_EXTENDS_TYPE = "http://www.gupro.de/GXL/gxl-1.0.gxl#isA"

def is_node_or_edge(node):
    return get_node_type(node) in [GXL_NODE_TYPE, GXL_EDGE_TYPE]

class GxlClass(object):
    def __init__(self, node):
        def getnodeattr(node, name, default=None):
            matches = [v for v in node.getElementsByTagName("attr")
                if v.getAttribute("name") == name]
            if not matches and default is not None:
                return default
            return get_gxl_value(get_singleton(matches))

        self.name = node.getAttribute("id")
        self.is_abstract = getnodeattr(node, "isabstract", False)
        self.typ = "node" if get_node_type(node) == GXL_NODE_TYPE else "edge"
        self.base_classes = []
        self.extending_classes = []

    def add_extends(self, other):
        self.base_classes.append(other)
        other.extending_classes.append(self)

    def __repr__(self):
        return "gxl-%s[extends=%s]" % (self.name, ", ".join(v.name for v in self.base_classes))

def elt_children(node):
    return [v for v in node.childNodes if v.nodeType == v.ELEMENT_NODE]

def get_gxl_value(node):
    node = get_singleton(elt_children(node))
    nv = node.firstChild.nodeValue
    if node.tagName == "bool":
        return nv.lower() in ["true", "yes", "1"]
    raise NotImplementedError, "type %s" % (node.tagName)

def get_node_type(node):
    type_node = get_singleton(v for v in elt_children(node) if v.tagName == "type")
    return type_node.getAttribute("xlink:href")




def elt_classes_by_id(x):
    elts_by_id = list(x).equiv_classes(lambda v: v.name)
    return dict(elts_by_id).map_values(get_singleton)

def generate_graph(elt_classes, extends_edges):
    elts_by_id = list(elt_classes).equiv_classes(lambda v: v.name)
    elts_by_id = dict(elts_by_id).map_values(get_singleton)
    for v in extends_edges:
        from_, to_ = v.getAttribute("from"), v.getAttribute("to")
        if not from_ in elts_by_id:
            warn("from node not present")
        elif not to_ in elts_by_id:
            warn("to node not present")
        else:
            elts_by_id[from_].add_extends(elts_by_id[to_])
    result = list(elt_classes).equiv_classes(lambda v: v.typ)
    return result["node"], result["edge"]

@memoize_file(".gen/python_graph_classes", use_hash=True)
def get_graph(text):
    doc = xml.parseString(text)
    type_graph = get_singleton(v for v in doc.getElementsByTagName("graph") if v.getAttribute("id") == "SCE_ScalaAstModel")
    gxl_nodes = type_graph.getElementsByTagName("node")
    gxl_edges = type_graph.getElementsByTagName("edge")
    # generate nicer objects and filter decls in the type graph
    elt_classes = [GxlClass(v) for v in gxl_nodes if is_node_or_edge(v)]
    extends_edges = [v for v in gxl_edges if get_node_type(v) == GXL_EXTENDS_TYPE]
    return generate_graph(elt_classes, extends_edges)

modpath = Path(__file__).parent(nmore=4)
print(modpath)

def main(show_typegraph=False):
    fname = get_singleton(v for v in modpath.subpath("plugin/src/main/resources").walk_files()
        if v.basename() == "type_graph.gxl")
    if show_typegraph:
        SubProc(["less", fname]).start_wait()
    return get_graph(fname.read())

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] args")
    cmdopts.add_option("--show_typegraph", action="store_true",
        help="show the typgraph gxl file in less for debugging")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)
