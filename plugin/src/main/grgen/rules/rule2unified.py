#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple, defaultdict
import sys; reload(sys); sys.setdefaultencoding('utf-8') # print unicode
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, persistent_var, pprint, process_jinja2, set, sort_asc,
    sort_desc)

def main(filename, name=None):
    filename = Path(filename)
    name = name if name else filename.basename().rstrip(".grg")
    assert filename.isfile() and filename.endswith(".grg")
    file2 = filename.parent(1).subpath("unified").subpath(name + ".unified.grg")
    assert not file2.exists(), "path %s exists!" %(file2)
    text = """{%% import "macros.grg" as macros with context %%}

{%% block comment %%}
// author: gatoatigrado (nicholas tung) [ntung at ntung]
// copyright: University of California, Berkeley
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain a
// copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
//
// DESCRIPTION OF FILE HERE
{%% endblock %%}

{%% block grg %%}
%s
{%% endblock %%}
""" %(filename.read())
    file2.write(text)
    SubProc(["gvim", str(file2)]).start()
    filename.unlink()

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options]",
        description="add blocks around a .grg file to convert it to a unified grg file")
    cmdopts.add_option("--name", help="set the name of the new file, if different")
    options, args = cmdopts.parse_args()
    if len(args) < 0:
        cmdopts.error("invalid number of arguments")
    main(*args, **options.__dict__)
