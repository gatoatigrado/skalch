#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

from __future__ import division
from collections import defaultdict, namedtuple
from net.sourceforge.gxl import (GXLAtomicValue, GXLAttr, GXLAttributedElement,
    GXLBool, GXLDocument, GXLEdge, GXLElement, GXLFloat, GXLGraph,
    GXLGraphElement, GXLGXL, GXLInt, GXLNode, GXLString, GXLType,
    GXLTypedElement, GXLValue)
from java.io import File
import re
from warnings import warn

class GraphElementWrapper(object):
    def __init__(self, elt):
        attrs = [elt.getAttrAt(i) for i in range(elt.getAttrCount())]
        attrs = dict([(v.name, v.value.value) for v in attrs])
        self.__dict__.update(attrs)

        self.id = elt.ID
        m = re.match("^#([\w\-\_]+)$", str(elt.getType().getURI()))
        assert m, "type didn't match"
        self.type = m.group(1)
        self.attrs = attrs
        self.edges = { }

    def set_edge(self, name, target):
        if name in self.edges and self.edges[name] != target:
            warn("overriding multiedge from %s to %s" %(self, target))
        self.edges[name] = target
        if hasattr(self, name):
            warn("attribute name '%s' conflicts; ignoring" %(name))
        else:
            setattr(self, name, target)

class GraphNodeWrapper(GraphElementWrapper): pass

class GraphEdgeWrapper(GraphElementWrapper):
    def __init__(self, edge):
        GraphElementWrapper.__init__(self, edge)
        self.sourceId = edge.sourceID
        self.targetId = edge.targetID

def wrap_elt(elt):
    if isinstance(elt, GXLNode):
        return GraphNodeWrapper(elt)
    elif isinstance(elt, GXLEdge):
        return GraphEdgeWrapper(elt)
    else:
        warn("unknown type for node %r" %(elt))

class GraphAbstraction(object):
    def __init__(self, graph):
        self.graph = graph
        self.elts = [wrap_elt(graph.getGraphElementAt(i))
            for i in range(graph.graphElementCount)]
        self.elts = [v for v in self.elts if v != None]
        self.by_type = defaultdict(list)
        self.by_attr = defaultdict(lambda: defaultdict(list))
        self.by_id = { }
        for elt in self.elts:
            assert not elt.id in self.by_id
            self.by_id[elt.id] = elt
            [self.by_attr[k][v].append(elt) for k, v in elt.attrs.items()]
            self.by_type[elt.type].append(elt)
        for elt in self.elts:
            if isinstance(elt, GraphEdgeWrapper):
                source = self.by_id[elt.sourceId]
                target = self.by_id[elt.targetId]
        bang_bang_calls = self.by_type["BangBangCall"]

def main(gxl_filename):
    import os
    assert os.path.isfile(gxl_filename)
    doc = GXLDocument(File(gxl_filename))
    gxl = doc.getDocumentElement()
    graphs = [gxl.getGraphAt(i) for i in range(gxl.getGraphCount())]
    graph = GraphAbstraction([g for g in graphs if g.getID() == "DefaultGraph"][0])

    packages = graph.by_type["PackageDef"]
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] gxl_file")
    # cmdopts.add_option("--myoption", help="help")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)
