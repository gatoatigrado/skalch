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

from parse_tag import *
import get_typegraph

# N.B. -- the goal is to write maybe 90% of the conversion code in this.
# Special cases should be handled by manual Java code.
GXL_TO_SKETCH = """
ClassDef(impl, params[], list)
    -> TypeStruct(<ctx>)
"""

def main():
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
        print("match(%s)" % (typname))
        if typename in rules:
            print("    rewrite rule set")
    print(rules)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] args")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)
