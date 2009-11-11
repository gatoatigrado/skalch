#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

from __future__ import division
from collections import defaultdict, namedtuple

from net.sourceforge.gxl import GXLEdge, GXLNode
import re
from warnings import warn

scala_to_sketch_types = {
    "scala.Boolean": "bit",
    "scala.Int": "int" }

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
        self.important_attrs = dict((k, v) for k, v in attrs.items()
            if k in ["symbolName"])
        #self.reserved_names = dir(self) + ["reserved_names"]

    def __repr__(self):
        return "%s[id=%s%s]" %(self.type, self.id, "".join(
            ", %s=%s" %(k, v) for k, v in self.important_attrs.items()))

class GraphNodeWrapper(GraphElementWrapper):
    reserved_name_lst = []

    def __init__(self, node):
        GraphElementWrapper.__init__(self, node)
        self.edges = { }

    def pprint(self):
        print(repr(self))
        for name, value_arr in self.edges.items():
            if len(value_arr) == 1:
                print("%s%s :   %s" %(4 * " ", name, value_arr[0]))
            arrname = name + ("s" if not name.endswith("s") else "_arr")
            print("%s%s :   %s" %(4 * " ", arrname, value_arr))

    def get_chain(self, name):
        return getattr(self, name + "Chain").get_chain_next(
            name + "Next")

    def get_chain_next(self, name):
        return [self] + (getattr(self, name).get_chain_next(name)
            if hasattr(self, name) else [])

    def set_edge(self, name, target):
        if name.startswith(self.type):
            name = name[len(self.type):]
        #assert not name in self.reserved_names
        #print("set edge %r -:%s-> %r" %(self, name, target))
        if name in self.edges:
            hasattr(self, name) and delattr(self, name)
            self.edges[name].append(target)
        else:
            assert not hasattr(self, name), \
                "name %s conflicts with class internals" %(name)
            self.edges[name] = [target]
            setattr(self, name, target)
            arrname = name + ("s" if not name.endswith("s") else "_arr")
            setattr(self, arrname, self.edges[name])

    def get_name(self):
        if hasattr(self, "name"):
            return self.name
        else:
            self.name = self.fullSymbolName.replace(".").replace("$", "D")
            self.name = (self.name + str(v) for v in range(100)
                if not (self.name + str(v)) in self.__class__.reserved_name_lst).next()
            return self.name

    def typename(self):
        if getattr(self, "fullSymbolName", None) in scala_to_sketch_types:
            return scala_to_sketch_types[self.fullSymbolName]
        elif self.type == "Symbol":
            return self.get_name()
        else:
            assert False, "typename() called on %r" %(self)

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
        self.edges = []
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
                elt.source = self.by_id[elt.sourceId]
                elt.target = self.by_id[elt.targetId]
                elt.source.set_edge(elt.type, elt.target)
                self.edges.append(elt)
        bang_bang_calls = self.by_type["BangBangCall"]
