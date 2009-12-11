#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple

try:
    from gatoatigrado_lib import (ExecuteIn, Path, SubProc, get_singleton,
        list, memoize_file, pprint, set, sort)
except:
    raise ImportError("please install gatoatigrado's utility library from "
            "bitbucket.org/gatoatigrado/gatoatigrado_lib")

from parse_tag import parse_gxl_conversion
import get_typegraph

# N.B. -- the goal is to write maybe 90% of the conversion code in this.
# Special cases should be handled by manual Java code.
GXL_TO_SKETCH = """
ClassDef(ClassDefSymbol, ClassDefFieldsList[].symbolName, 
        ClassDefFieldsList[]:TypeSymbol:SketchType)

    -> TypeStruct(<ctx>, String, List[String], List[Type])
"""

def get_node_match_cases():
    rules = { }
    for rule in parse_gxl_conversion(GXL_TO_SKETCH).argv:
        assert not str(rule.gxlname) in rules
        rules[str(rule.gxlname)] = rule
    node_types, edge_types = get_typegraph.main(show_typegraph=False)
    node_types = get_typegraph.elt_classes_by_id(node_types)
    edge_types = get_typegraph.elt_classes_by_id(edge_types)

    node_match_cases = []
    def genMatchCases(node_type):
        [genMatchCases(v) for v in node_type.extending_classes]
        if node_type.name in rules:
            node_match_cases.append(rules[node_type.name])
    genMatchCases(node_types["Node"])
    return node_match_cases

def ast_inheritance():
    immediate = {
#        "Object": "Type Class String",
#        "Class": "ExprBinary",
        "Type": "TypeStruct TypePrimitive" }

    immediate = dict( (k, v.split()) for k, v in immediate.items() )

    def get_lowest_arr(v):
        if v in immediate:
            return list(set(final_type for sub_type in immediate[v] for final_type in get_lowest_arr(sub_type)))
        else:
            return [v]

    return dict( (k, get_lowest_arr(k)) for k in immediate.keys() )
