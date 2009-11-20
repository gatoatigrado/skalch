#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

from __future__ import division, print_function
from collections import namedtuple
import re
from jinja2 import Environment, FileSystemLoader
import jinja2.ext

try:
    # Path, ExecuteIn, SubProc, sort()
    from gatoatigrado_lib import *
except:
    import sys; print("please install gatoatigrado's utility library from "
        "bitbucket.org/gatoatigrado/gatoatigrado_lib", file=sys.stderr)

from generate_jinja2 import generate_jinja2

class FnameMod(object):
    def modify_cmd(self, cmd, fname):
        return "cat '%s' | %s" %(Path(fname), cmd)

class CdMod(object):
    def modify_cmd(self, cmd, fname):
        return "cd '%s'; %s" %(Path(fname).parent(), cmd)

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
        "of 'generate command' / 'begin generate block' / 'end generate block'."

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
                %("\n".join(outlines), "\n".join(block)))
            import sys; sys.exit(1)

def process_unified():
    grgen = Path("plugin/src/main/grgen")
    unified = grgen.subpath("unified")
    for unified_file in unified.walk_files():
        env = Environment(loader=FileSystemLoader(unified), trim_blocks=True,
            extensions=[jinja2.ext.do])
        #env.filters["pipetonull"] = lambda a: ""
        def get_output(blockname):
            all_names = "gm grg grs comment".split()
            other_blocks = [v for v in all_names if not v == blockname]
            basename = unified_file.relpath(unified)
            return env.from_string("{% extends '" + basename + "' %}\n" +
                 "\n".join("{%% block %s %%}{%% endblock %%}"
                %(v) for v in other_blocks)).render()

        name = unified_file.basename().replace(".unified.grg", "")
        paths = [grgen.subpath(d, "gen").makedirs().subpath(name + "." + ext)
            for d, ext in ( ("rules", "grg"), ("stages-scripts", "grs"), ("nodes", "gm") )]
        contents = [get_output(v) for v in "grg grs gm".split()]
        for d, ext in ( ("rules", "grg"), ("stages-scripts", "grs"), ("nodes", "gm") ):
            path = grgen.subpath(d, "gen").makedirs().subpath(name + "." + ext)
            if path.write_if_different(get_output(ext)):
                print("    generating '%s'" %(path))

def process_jinja2(files):
    for fname in files:
        env = Environment(loader=FileSystemLoader(fname.parent()),
            trim_blocks=True, extensions=[jinja2.ext.do])
        result = env.get_template(fname.basename()).render()
        if Path(fname.replace(".jinja2", "")).write_if_different(result):
            print(" " * 4 + "generating '%s'" %(fname))

@memoize_file(".generate_files.txt")
def files_with_generate():
    return list(SubProc(["grep", "-l", "@ generate command", "-R", "."]).exec_lines())

@memoize_file(".jinja2_file_list.txt")
def jinja2_file_list():
    return [v for v in Path(".").walk_files() if v.endswith(".jinja2")]

def main(no_rebuild=False,
        no_save_list=False, file_list_name=".generate_files.txt"):

    process_unified()
    files_with_generate.filename = file_list_name
    if not no_rebuild:
        files_with_generate.delete()
        jinja2_file_list.delete()
    lines = files_with_generate()
    process_jinja2(jinja2_file_list())
    return

    for fname in [Path(v) for v in lines]:
        if fname == Path(__file__):
            continue
        text = fname.read()
        stripped_lines = [v.strip() for v in text.split("\n")]
        check_cmdline_generated(fname, stripped_lines)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser(usage="%prog [options] args")
    cmdopts.add_option("--no_rebuild", action="store_true",
        help="don't rebuild the list of files with generate statements")
    cmdopts.add_option("--no_save_list", action="store_true",
        help="don't save the list of files with generate statements")
    options, args = cmdopts.parse_args()
    main(*args, **options.__dict__)

