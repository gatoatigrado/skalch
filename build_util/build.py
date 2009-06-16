#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys
import path_resolv
import custom_compile

compilers = ["fsc", "scalac", "javap"]
options = ["print_lowered"]
defaultopts = { "fsc": "-classpath %(classpath)s -sourcepath %(src_path)s".split(" "),
    "javap": "-classpath %(classpath)s:%(out_path)s".split(" ") }
defaultopts["scalac"] = defaultopts["fsc"]

def set_classpath_from_eclipse():
    from xml.dom.minidom import parse as parse_xml
    eclipse_classpath = path_resolv.Path(".classpath")
    if not eclipse_classpath.exists():
        print("WARNING - please run with \".classpath\" in $(pwd), attempting to resolve...")
        eclipse_classpath = path_resolv.resolve(".classpath")
        assert eclipse_classpath.exists()
    doc = parse_xml(eclipse_classpath)
    entries = doc.getElementsByTagName("classpathentry")
    path_elts = [os.environ["CLASSPATH"]]
    for entry in entries:
        if entry.getAttribute("kind") == "lib":
            path_elts.append(path_resolv.resolve(entry.getAttribute("path")))
    os.environ["CLASSPATH"] = path_resolv.Path.pathjoin(*path_elts)

if __name__ == "__main__":
    set_classpath_from_eclipse()
    in_args = sys.argv[1:]
    run_after = False

    if not in_args:
        print("compiling all; use 'help' to show help")
    if "help" in in_args or any(["-h" in arg for arg in in_args]):
        print(r"""usage: build.py [compiler] [options]
compilers (one chosen):
    %r
expanded options / actions (use as many as you want):
    %r
    /filter where filter is a regular expression for source files to compile
all other options passed to compiler. common ones:
    ['clean', 'src_java_names', 'run']""" %(compilers, options), file=sys.stderr)
        sys.exit(1)

    compiler = "scalac"
    if in_args and in_args[0] in compilers:
        compiler = in_args[0]
        in_args = in_args[1:]
    compile_args = ["--compiler=%s" %(compiler)]
    [compile_args.extend(["--option", opt]) for opt in defaultopts[compiler]]

    run_options = False
    for arg in in_args:
        if arg == "print_lowered":
            compile_args.extend(["--option", "-Xprint:jvm"])
        elif arg.startswith("/"):
            compile_args.append("--src_filter=%s" %(arg[1:]))
        elif arg == "--run_opt_list":
            run_options = True
        elif run_options:
            compile_args.extend(["--run_option", arg])
        else:
            compile_args.append("--%s" %(arg.lstrip("-")))

    custom_compile.main(compile_args)
