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

from parse_tag import parse_gxl_conversion, GxlSubtree, GxlSubtreeOL, \
    GxlSubtreeUL
import get_typegraph

# N.B. -- the goal is to write maybe 90% of the conversion code in this.
# Special cases should be handled by manual Java code.
STRUCTURE = r"""
PackageDef(<this>, UL[PackageDefElement])
    -> new Program(<ctxnode>, SingletonList[StreamSpec], List[TypeStruct])

PackageDef(UL[PackageDefGlobal], UL[PackageDefFcn])
    -> new StreamSpec(<ctx>, "StreamSpec.STREAM_FILTER", "new StreamType((FEContext)null,
            TypePrimitive.bittype, TypePrimitive.bittype)", "\"MAIN\"", "Collections.EMPTY_LIST", List[FieldDecl], List[Function])

ClassDef(ClassDefSymbol:PrintSymName, OL[ClassDefFieldsList]:PrintSymName,
        OL[ClassDefFieldsList]:TypeSymbol:SketchType)
    -> new TypeStruct(<ctx>, String, List[String], List[Type])

FcnDef(FcnDefSymbol:PrintSymName, FcnDefReturnTypeSymbol, OL[FcnDefParamsList], FcnBody, FcnDefIsGenerator.value)
    -> Function(<ctx>, String, Type, List[Parameter], "getImplements(node)", Statement, boolean)
"""



STMTS = r"""
ValDef(ValDefSymbol, ValDefSymbol:PrintSymName)
    -> new StmtVarDecl(<ctx>, Type, String, <null>)

ValDef(ValDefSymbol, ValDefSymbol:PrintSymName, SketchParamType.typecode)
    -> new Parameter(Type, String, int)

ValDef(ValDefSymbol, ValDefSymbol:PrintSymName)
    -> new FieldDecl(<ctxnode>, Type, String, "null")

VarRef(VarRefSymbol:PrintSymName)
    -> new ExprVar(<ctx>, String)

SKAssertCall(SKAssertCallArg)
    -> new StmtAssert(<ctx>, Expression, "false")

SKBlock(OL[BlockStmtList])
    -> new StmtBlock(<ctx>, List[Statement])

SKStmtExpr(SKStmtExprExpr)
    -> new StmtExpr(<ctxnode>, Expression)

Return(ReturnExpr)
    -> new StmtReturn(<ctx>, Expression)

Assign(AssignLhs, AssignRhs)
    -> new StmtAssign(<ctxnode>, Expression, Expression, "0")

If(IfCond, IfThen, IfElse)
    -> new StmtIfThen(<ctx>, Expression, Statement, Statement)

SKWhileLoop(SKWhileLoopCond, SKWhileLoopBody)
    -> new StmtWhile(<ctx>, Expression, Statement)

SyncthreadsCall()
    -> new CudaSyncthreads(<ctx>)

UnitConstant()
    -> new StmtEmpty(<ctx>)
"""



#FieldAccess(FieldAccessObject, ...)
#    -> new ExprField(<ctxnode>, Expression,
EXPRS = r"""
FcnBinaryCall(FcnBinaryCallLhs, .strop, FcnBinaryCallRhs)
    -> new ExprBinary(<ctxnode>, Expression, String, Expression)

FcnCallUnaryNegative(OL[FcnArgList])
    -> new ExprUnary(<ctx>, "ExprUnary.UNOP_NEG", GetListSingleton[Expression])

FcnCall(FcnCallSymbol:PrintSymName, OL[FcnArgList])
    -> new ExprFunCall(<ctx>, String, List[Expression])

SKNew(FcnCallTypeSymbol)
    -> new ExprNew(<ctx>, Type)

FieldAccess(FieldAccessObject, FieldAccessSymbol:PrintSymName)
    -> new ExprField(<ctxnode>, Expression, String)

HoleCall(FcnCallTypeSymbol) -> new ExprStar(<ctx>, "4", Type)

NullTypeConstant() -> new ExprNullPtr(<ctx>)

BooleanConstant(.value) -> new ExprConstBoolean(<ctx>, boolean)

IntConstant(.value) -> new ExprConstInt(<ctx>, int)

UnitConstant() -> new ExprConstUnit(<ctx>)

NewArray(OL[ArrValueList]) -> new ExprArrayInit(<ctx>, List[Expression])

SketchArrayAccess(SketchArrayAccessArray, SketchArrayAccessIndex)
    -> new ExprArrayRange(<ctxnode>, Expression, Expression)

SketchTprintCall(OL[PrintCallArgList])
    -> new ExprTprint(<ctx>, List[TprintTuple])

SketchPrintTuple(SketchPrintTupleName.value, SketchPrintTupleValue)
    -> new TprintTuple(String, Expression)

SketchThreadIdx(.indexName)
    -> new CudaThreadIdx(<ctx>, String)

SketchBlockDim(.indexName)
    -> new CudaBlockDim(<ctx>, String)
"""



TYPES = r"""
Symbol(SketchType, TermMemLocationType)
    -> Type(Type, CudaMemoryType)
TypeBoolean() -> TypePrimitive "TypePrimitive.bittype"
TypeInt() -> TypePrimitive "TypePrimitive.inttype"
TypeUnit() -> TypePrimitive "TypePrimitive.voidtype"
TypeArray(ArrayInnerTypeSymbol, ArrayLengthExpr) -> new TypeArray(Type, Expression)

TypeStructRef() -> new TypeStructRef("createString(getStringAttribute(\"typename\", node))")
"""

MISC = r"""
PrintName(.name) -> String(String)

CudaMemShared() -> CudaMemoryType "CudaMemoryType.GLOBAL"
CudaMemImplicitShared() -> CudaMemoryType "CudaMemoryType.GLOBAL"
CudaMemDefaultShared() -> CudaMemoryType "CudaMemoryType.GLOBAL"
CudaMemGlobal() -> CudaMemoryType "CudaMemoryType.GLOBAL"
CudaMemLocal() -> CudaMemoryType "CudaMemoryType.LOCAL"
"""

GXL_TO_SKETCH = STRUCTURE + STMTS + EXPRS + TYPES + MISC

# FENode context, int cls, String name, Type returnType, List<Parameter> params, Statement body

def get_node_match_cases():
    rules = parse_gxl_conversion(GXL_TO_SKETCH).argv

    node_types, edge_types = get_typegraph.main(show_typegraph=False)
    node_types = get_typegraph.elt_classes_by_id(node_types)
    edge_types = get_typegraph.elt_classes_by_id(edge_types)

    for rule in rules:
        if not unicode(rule.gxlname) in node_types:
            warn("unknown gxl node '%s' (maybe try updating the type graph?)" % (rule.gxlname))
        for subfield in [v for arg in rule.gxl_args for v in arg]:
            if isinstance(subfield, (GxlSubtree, GxlSubtreeOL, GxlSubtreeUL)):
                if not unicode(subfield.name.text) in edge_types:
                    warn("unknown edge type '%s'" % (subfield.name.text))

    return rules

@memoize_file(".gen/sketch_fe_ast_node_types")
def get_java_ast_node_types(fe_directory):
    return ""

def ast_inheritance(rules):
    # the immediate inheritance are objects directly below a given object
    immediate = {
#        "Object": "Type Class String",
#        "Class": "ExprBinary",
        "Expression": "ExprBinary ExprStar ExprConstant ExprVar ExprUnary ExprFunCall ExprField ExprNullPtr ExprNew ExprArrayInit ExprArrayRange ExprTprint CudaThreadIdx CudaBlockDim",
        "ExprConstant": "ExprConstBoolean ExprConstInt ExprConstUnit",
        "Statement": "StmtVarDecl StmtAssert StmtBlock StmtReturn StmtAssign StmtIfThen StmtExpr StmtWhile CudaSyncthreads StmtEmpty",
        "Type": "TypeStruct TypePrimitive TypeStructRef TypeArray" }
    immediate = dict((k, v.split()) for k, v in immediate.items())
    for rule in rules:
        name = str(rule.javaname)
        if not name in immediate:
            immediate[name] = ""
    assert all(type(v) == str for v in immediate.keys())

    exact_typ_rules = dict((k, [rule for rule in rules if str(rule.javaname) == k]) for k in immediate.keys())
    assert all(all(v in immediate for v in arr) for arr in immediate.values()), \
        "some types in the RHS of immediate { } aren't defined."

    def buildchain(result_typ):
        # build up in reverse -- last items will be most specific, first will be most generic
        checks = exact_typ_rules[result_typ] # start out with most generic
        expand_queue = immediate[result_typ]
        while expand_queue:
            checks += [v for to_expand in expand_queue for v in exact_typ_rules[to_expand]]
            expand_queue = [v for to_expand in expand_queue for v in immediate[to_expand]]
        checks = [ (str(v.gxlname), "get%sFrom%s" % (v.javaname, v.gxlname)) for v in checks ]
        return checks[::-1]

    return dict((k, buildchain(k)) for k in immediate.keys())
