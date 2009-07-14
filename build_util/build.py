#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, subprocess, sys
import path_resolv
import custom_compile

pathsep = os.path.pathsep

# NOTE - unfortunately, -optimise doesn't terminate on RegexGen.
defaultopts = { "fsc": "-classpath %(classpath)s -sourcepath %(src_path)s -Xplugin:%(plugin_path)s".split(" "),
    "javac": "-classpath %(classpath)s -sourcepath %(src_path)s -d %(out_path)s".split(" "),
    "javap": ("-classpath %(classpath)s" + os.path.pathsep +  "%(out_path)s").split(" ") }
defaultopts["scalac"] = defaultopts["fsc"]
for k, v in defaultopts.items():
    defaultopts[k + ".bat"] = v
compilers = defaultopts.keys()

os.environ["CLASSPATH"] = path_resolv.Path.pathjoin(
    os.environ["CLASSPATH"],
    path_resolv.resolve("lib/gnu-regexp-1.1.4.jar"),
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

usage_text = r"""usage: build.py [%s] [actions] [options] [/filter]
    [run_app=qualified.name
        [run_cmd_opts <options to java>]
        [run_opt_list <opts to program>]]

build.py actions:
    print_lowered       alias for --option -Xprint:jvm
    no_default_opts     don't use default options
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
%s""" %(" | ".join(compilers), custom_compile.get_options_help())

def print_help():
    try:
        proc = subprocess.Popen(["less"], stdin=subprocess.PIPE)
        proc.communicate(usage_text)
    except:
        print(usage_text)
    sys.exit(1)

def main(in_args):
    set_classpath_from_eclipse()
    compiler = "scalac"
    if in_args and in_args[0] in compilers:
        compiler = in_args[0]
        in_args = in_args[1:]
    compile_args = ["--compiler=%s" %(compiler)]
    if compiler == "javac":
        compile_args.extend(["--src_extensions", "java"])

    mode = "default"
    usedefaultopts = True
    for arg in in_args:
        arg_trimmed = arg.lstrip("-")

        if arg.startswith("/"):
            compile_args.append("--src_filter=%s" %(arg[1:]))
            
        # change the mode
        elif arg_trimmed == "run_opt_list":
            mode = "run options"
        elif arg_trimmed == "run_cmd_opts":
            mode = "java options"

        # argument lists
        elif mode == "run options":
            compile_args.extend(["--run_option", arg])
        elif mode == "java options":
            compile_args.extend(["--java_option", arg])

        elif arg_trimmed == "print_lowered":
            compile_args.extend(["--option", "-Xprint:jvm"])
        elif arg_trimmed == "no_default_opts":
            usedefaultopts = False

        # help
        elif arg == "-h" or arg_trimmed == "help":
            print_help()
        else:
            compile_args.append("--%s" %(arg.lstrip("-")))

    if usedefaultopts:
        [compile_args.extend(["--option", opt]) for opt in defaultopts[compiler]]

    custom_compile.main(compile_args)

if __name__ == "__main__":
    main(sys.argv[1:])
