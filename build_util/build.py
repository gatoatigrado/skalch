#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, subprocess, sys
import path_resolv
import custom_compile

options = ["print_lowered"]
defaultopts = { "fsc": "-classpath %(classpath)s -sourcepath %(src_path)s".split(" "),
    "javac": "-classpath %(classpath)s -sourcepath %(src_path)s -d %(out_path)s".split(" "),
    "javap": "-classpath %(classpath)s:%(out_path)s".split(" ") }
defaultopts["scalac"] = defaultopts["fsc"]
compilers = defaultopts.keys()

os.environ["CLASSPATH"] = path_resolv.Path.pathjoin(
    os.environ["CLASSPATH"],
    path_resolv.resolve("lib/swing-layout-1.0.3.jar"))
# reinit with updated classpath
path_resolv.resolvers[0] = path_resolv.EnvironPathResolver()

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

def print_help():
    proc = subprocess.Popen(["less"], stdin=subprocess.PIPE)
    proc.communicate(r"""usage: build.py [%s] [actions] [options] [/filter]
    [run_app=qualified.name
        [run_cmd_opts <options to java>]
        [run_opt_list <opts to program>]]

build.py actions: %r
    /filter where filter is a regular expression for source files to compile




=== examples ===

    build scala code and run RomanNumeralsTest:
        build_util/build.py print run_app=test.RomanNumeralsTest

    run BitonicSortTest with argument -h passed to java:
        build_util/build.py /dontcompile run_app=test.BitonicSortTest --run_opt_list -h

    if fsc happens to be working for you:
        build_util/build.py fsc print run_app=test.RomanNumeralsTest



=== options to custom_compile.py ===

all other options passed to compiler. common ones: include [clean | src_java_names]
%s""" %(" | ".join(compilers), options, custom_compile.get_options_help()))
    sys.exit(1)

if __name__ == "__main__":
    set_classpath_from_eclipse()
    in_args = sys.argv[1:]
    run_after = False

    compiler = "scalac"
    if in_args and in_args[0] in compilers:
        compiler = in_args[0]
        in_args = in_args[1:]
    compile_args = ["--compiler=%s" %(compiler)]
    [compile_args.extend(["--option", opt]) for opt in defaultopts[compiler]]
    if compiler == "javac":
        compile_args.extend(["--src_extensions", "java"])

    mode = "default"
    for arg in in_args:
        if arg == "print_lowered":
            compile_args.extend(["--option", "-Xprint:jvm"])
        elif arg.startswith("/"):
            compile_args.append("--src_filter=%s" %(arg[1:]))

        # argument lists
        elif arg.lstrip("-") == "run_opt_list":
            mode = "run options"
        elif arg.lstrip("-") == "run_cmd_opts":
            mode = "java options"
        elif mode == "run options":
            compile_args.extend(["--run_option", arg])
        elif mode == "java options":
            compile_args.extend(["--java_option", arg])

        # help
        elif arg == "-h" or arg.strip("-") == "help":
            print_help()
        else:
            compile_args.append("--%s" %(arg.lstrip("-")))

    custom_compile.main(compile_args)
