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

try:
    from gatoatigrado_lib import (ExecuteIn, Path, SubProc, get_singleton, dict,
        list, memoize_file, pprint, set, sort)
except:
    raise ImportError("please install gatoatigrado's utility library from "
            "bitbucket.org/gatoatigrado/gatoatigrado_lib")

from parse_tag import parse_gxl_conversion, GxlSubtree, GxlSubtreeOL,\
    GxlSubtreeUL
import get_typegraph

# N.B. -- the goal is to write maybe 90% of the conversion code in this.
# Special cases should be handled by manual Java code.
GXL_TO_SKETCH = r"""
PackageDef(<this>, UL[PackageDefElement])
    -> new Program(<ctxnode>, SingletonList[StreamSpec], List[TypeStruct])

PackageDef(UL[PackageDefGlobal], UL[PackageDefFcn])
    -> new StreamSpec(<ctx>, "StreamSpec.STREAM_FILTER", "new StreamType((FEContext)null,
            TypePrimitive.bittype, TypePrimitive.bittype)", "\"MAIN\"", "Collections.EMPTY_LIST", List[StmtVarDecl], List[Function])

ClassDef(ClassDefSymbol, OL[ClassDefFieldsList].symbolName, 
        OL[ClassDefFieldsList]:TypeSymbol:SketchType)
    -> new TypeStruct(<ctx>, String, List[String], List[Type])

ValDef(ValDefSymbol:TypeSymbol, ValDefSymbol.symbolName)
    -> new StmtVarDecl(<ctx>, Type, String, <null>)

ValDef(ValDefSymbol:TypeSymbol, ValDefSymbol.symbolName)
    -> new Parameter(Type, String)

SKAssertCall(FcnArgList)
    -> new StmtAssert(<ctx>, Expression, "false")

FcnDef(FcnDefSymbol.symbolName, FcnDefReturnTypeSymbol, OL[FcnDefParamsList], FcnBody)
    -> new Function(<ctx>, "Function.FUNC_WORK", String, Type, List[Parameter], Statement)
"""

# FENode context, int cls, String name, Type returnType, List<Parameter> params, Statement body

def get_node_match_cases():
    rules = list(parse_gxl_conversion(GXL_TO_SKETCH).argv).equiv_classes(
        lambda a: str(a.javaname))

    node_types, edge_types = get_typegraph.main(show_typegraph=False)
    node_types = get_typegraph.elt_classes_by_id(node_types)
    edge_types = get_typegraph.elt_classes_by_id(edge_types)

    for rule in [v for arr in rules.values() for v in arr]:
        if not unicode(rule.gxlname) in node_types:
            warn("unknown gxl node '%s' (maybe try updating the type graph?)" %(rule.gxlname))
        for subfield in [v for arg in rule.gxl_args for v in arg]:
            if isinstance(subfield, (GxlSubtree, GxlSubtreeOL, GxlSubtreeUL)):
                if not unicode(subfield.name.text) in edge_types:
                    warn("unknown edge type '%s'" %(subfield.name.text))

    def sort_types(arr):
        rules_for_javaname = dict((str(v.gxlname), v) for v in arr)
        node_match_cases = []
        def genMatchCases(node_type):
            [genMatchCases(v) for v in node_type.extending_classes]
            if node_type.name in rules_for_javaname:
                node_match_cases.append(rules_for_javaname[node_type.name])
        genMatchCases(node_types["Node"])
        return node_match_cases

    return dict(rules).map_values(sort_types)

@memoize_file(".gen/sketch_fe_ast_node_types")
def get_java_ast_node_types(fe_directory):
    return ""

def ast_inheritance():
    immediate = {
#        "Object": "Type Class String",
#        "Class": "ExprBinary",
        "Expression": "ExprField",
        "Statement": "StmtVarDecl StmtAssert",
        "Type": "TypeStruct TypePrimitive" }

    immediate = dict((k, v.split()) for k, v in immediate.items())

    def get_lowest_arr(v):
        if v in immediate:
            return list(set(final_type for sub_type in immediate[v] for final_type in get_lowest_arr(sub_type)))
        else:
            return [v]

    return dict((k, get_lowest_arr(k)) for k in immediate.keys())
