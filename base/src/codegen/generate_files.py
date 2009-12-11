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
    from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
        process_jinja2, list, memoize_file, pprint, set, sort)
except:
    raise ImportError("please install gatoatigrado's utility library from "
            "bitbucket.org/gatoatigrado/gatoatigrado_lib")

from jinja2 import Environment, FileSystemLoader
import re, jinja2.ext
from gxltosketch.gxltosketch import get_node_match_cases, ast_inheritance
from gxltosketch import get_typegraph

class FnameMod(object):
    def modify_cmd(self, cmd, fname):
        return "cat '%s' | %s" % (Path(fname), cmd)

class CdMod(object):
    def modify_cmd(self, cmd, fname):
        return "cd '%s'; %s" % (Path(fname).parent(), cmd)

def get_modifiers(v):
    result = []
    if "run in dir of this file" in v:
        result.append(CdMod())
    elif "run on this file" in v:
        result.append(FnameMod())
    return result

def check_cmdline_generated(fname, stripped_lines):
    def filter_fcn(fcn):
        return [i for i, v in enumerate(stripped_lines) if fcn(v)]
    cmds = filter_fcn(lambda a: "@ generate command" in a)
    begins = filter_fcn(lambda a: "@ begin generate block" in a)
    ends = filter_fcn(lambda a: "@ end generate block" in a)
    assert len(cmds) == len(begins) == len(ends), "unmatching number " \
        "of 'generate command' / 'begin generate block' / 'end generate block'." \
        "for file %s" % (fname)

    # existing data
    commands = [stripped_lines[(a + 1):b] for a, b in zip(cmds, begins)]
    modifiers = [get_modifiers(stripped_lines[v]) for v in cmds]
    proc = lambda v: re.sub(r"\\$", "", re.sub(r"^(// |# |\*/)", "", v))
    commands = [" ".join([proc(v) for v in cmd]) for cmd in commands]
    blocks = [stripped_lines[(a + 1):b] for a, b in zip(begins, ends)]

    # check blocks
    for cmd, mods, block in zip(commands, modifiers, blocks):
        for mod in mods:
            cmd = mod.modify_cmd(cmd, fname)
        outlines = list(SubProc(["zsh", "-c", cmd]).exec_lines())
        outlines = [v.strip() for v in outlines]
        if outlines != block:
            print("Differing lines:\nGenerated:\n\n%s\n\nCurrent:\n\n%s"
                % ("\n".join(outlines), "\n".join(block)))
            import sys; sys.exit(1)

def process_unified():
    grgen = Path("plugin/src/main/grgen")
    unified = grgen.subpath("unified")
    env = Environment(loader=FileSystemLoader(unified), trim_blocks=True,
        extensions=[jinja2.ext.do])

    @memoize_file(".gen/unified_processed_results", use_hash=True)
    def process_file_inner(fname, text_hash):
        print("processing changed file", fname.basename())
        def get_output(blockname):
            all_names = "gm grg grs comment".split()
            other_blocks = [v for v in all_names if not v == blockname]
            basename = fname.relpath(unified)
            return env.from_string("{% extends '" + basename + "' %}\n" +
                 "\n".join("{%% block %s %%}{%% endblock %%}"
                % (v) for v in other_blocks)).render()

        name = fname.basename().replace(".unified.grg", "")
        result = []
        for d, ext in (("rules", "grg"), ("stages-scripts", "grs"), ("nodes", "gm")):
            path = grgen.subpath(d, "gen").makedirs().subpath(name + "." + ext)
            result.append((path, get_output(ext)))
        return result

    def process_file(fname):
        for path, output in process_file_inner(fname, hash(str(fname.read()))):
            if path.write_if_different(output):
                print("    generating '%s'" % (path))

    [process_file(fname) for fname in unified.walk_files()
        if ".unified" in fname]

@memoize_file(".gen/generate_files.txt")
def files_with_generate():
    return list(SubProc(["grep", "-l", "@ generate command", "-R", "."]).exec_lines())

@memoize_file(".gen/jinja2_file_list.txt")
def jinja2_file_list():
    return [v for v in Path(".").walk_files() if v.endswith(".jinja2")]

generate_files_name = Path(__file__)
if "cProfile" in generate_files_name:
    generate_files_name = get_singleton(v
        for v in Path(".").walk_files() if v.endswith("generate_files.py"))

def main(single_file=None, no_rebuild=False):
    def assert_fcn(a, msg):
        assert a, msg
        return ""
    
    def print_fcn(*argv):
        import sys
        print(*argv, file=sys.stderr)
        sys.stderr.flush()

    class WrapperVariable(object):
        def __init__(self, value):
            self.value = value

        def setValue(self, value):
            self.value = value

    jinja2_glbls = { "enumerate": enumerate, "len": len, "assert": assert_fcn,
        "WrapperVariable": WrapperVariable, "print": print_fcn }

    resources_path = get_typegraph.modpath.subpath("plugin/src/main/resources")
    if any(v.read().strip() for v in resources_path.walk_files() if v.basename() == "type_graph.gxl"):
            jinja2_glbls["node_match_cases"] = get_node_match_cases()
            jinja2_glbls["ast_inheritance"] = ast_inheritance()

    # these are necessary for single file mode for the AST, so process it first.
    process_unified()
    if single_file:
        return process_jinja2(files=[Path(single_file)], glbls=jinja2_glbls)
    if not no_rebuild:
        files_with_generate.delete()
        jinja2_file_list.delete()
    lines = files_with_generate()
    process_jinja2(files=jinja2_file_list(), glbls=jinja2_glbls)

    for fname in [Path(v) for v in lines]:
        if fname == generate_files_name:
            continue
        text = fname.read()
        stripped_lines = [v.strip() for v in text.split("\n")]
        check_cmdline_generated(fname, stripped_lines)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] <filename>")
    cmdopts.add_option("--no_rebuild", action="store_true",
        help="don't rebuild the list of files with generate statements")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)

