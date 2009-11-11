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
from graph_abstraction import *

def print_struct(cls):
    pass

def print_methods(cls):
    impl = cls.Impl
    fcns = [v for v in impl.Elements if v.type == "MemberFcn"]
    for fcn in fcns:
        typename = fcn.FcnDefReturnTypeSymbol.typename()
        args = fcn.get_chain("FcnDefParams")
        v = args[0].Symbol
        #print("
        ident = 4 * " "
        body = fcn.FcnBody
        import pdb; pdb.set_trace()

def main(gxl_filename):
    import os
    assert os.path.isfile(gxl_filename)
    doc = GXLDocument(File(gxl_filename))
    gxl = doc.getDocumentElement()
    graphs = [gxl.getGraphAt(i) for i in range(gxl.getGraphCount())]
    graph = GraphAbstraction([g for g in graphs if g.getID() == "DefaultGraph"][0])

    packages = graph.by_type["PackageDef"]
    for package in packages:
        print("\n\n\n=== package %s ===\n" %(
            package.Symbol.fullSymbolName))
        map(print_struct, package.Elements)
        map(print_methods, package.Elements)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] gxl_file")
    # cmdopts.add_option("--myoption", help="help")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)
