#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple, defaultdict
import sys; reload(sys); sys.setdefaultencoding('utf-8') # print unicode
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, persistent_var, pprint, process_jinja2, set, sort_asc,
    sort_desc)
import re

scalaSourceRe1 = re.compile(r"scalaSource = \"[^\"]+\",", re.S)
scalaSourceRe2 = re.compile(r"scalaSource = \'[^\']+\',", re.S)

def main(fname):
    text = Path(fname).read()
    nospace = lambda a: a.group(0).replace("\n", "")
    text = scalaSourceRe1.sub(nospace, text)
    text = scalaSourceRe2.sub(nospace, text)
    Path(fname).write(text)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] fname",
        description="description")
    # cmdopts.add_option("--myflag", action="store_true", help="set my flag variable")
    options, args = cmdopts.parse_args()
    if len(args) < 1:
        cmdopts.error("invalid number of arguments")
    main(*args, **options.__dict__)
