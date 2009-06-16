#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import custom_compile

compilers = ["fsc", "scalac", "javap"]
options = ["print_lowered"]
defaultopts = { "fsc": "-classpath %(classpath)s -sourcepath %(src_path)s".split(" "),
    "javap": "-classpath %(classpath)s:%(out_path)s".split(" ") }
defaultopts["scalac"] = defaultopts["fsc"]

if __name__ == "__main__":
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

    for arg in in_args:
        if arg == "print_lowered":
            compile_args.extend(["--option", "-Xprint:jvm"])
        elif arg.startswith("/"):
            compile_args.append("--src_filter=%s" %(arg[1:]))
        else:
            compile_args.append("--%s" %(arg.lstrip("-")))

    custom_compile.main(compile_args)
