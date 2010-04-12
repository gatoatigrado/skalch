#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2010 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, persistent_var, pprint, process_jinja2, set, sort_asc,
    sort_desc)
from gatoatigrado_lib.subproc import ProcessException
import os
import re

@memoize_file("scala-tmp-srcdirs")
def get_srcdirs(file_list):
    file_list = file_list[:]
    while file_list:
        fname = file_list[0]
        text = fname.read()
        if "package" in text:
            print("normal...")
        else:
            print("bad...")

sketch_jars = list(Path("~/.m2/repository/edu/berkeley").walk_files(["jar"]))
def get_latest_jar(name):
    jars = sort_desc([v for v in sketch_jars if name in v])
    if not jars:
        import ipdb; ipdb.set_trace()
    assert jars, "couldn't find sketch project dependency '%s' in ~/.m2; try building with Maven" % (name)
    return jars[0]

@memoize_file("get-class-names")
def get_class_names(fname, modified_):
    return re.findall("(?:class|interface) ([\w\d\_]+)", fname.read())

# whatever's not actually rebuilt -- assume it's not actually a class name,
# since I don't want to parse java/scala sources
cls_name_exceptions = []
persistent_var(globals(), "cls_name_exceptions") # stored in python persistent vars

def is_up_to_date(srcfile, cls_files):
    def get_best_match(name):
        for fname in cls_files:
            if ("/" + name) in fname or ("$" + name) in fname:
                return fname
    
    clsnames = get_class_names(srcfile, srcfile.modified())
    clsnames = [v for v in clsnames if not v in cls_name_exceptions]
    cls_files_for_src = [get_best_match(v) for v in clsnames]
    missing_files = not all(cls_files_for_src)
    if not clsnames:
        print("no (non-excluded) class names for", srcfile, "compiling")
        return False
    if missing_files:
        return False
    if all(v.modified() > srcfile.modified() for v in cls_files_for_src):
        return True

def run_compile(classpath, outpath, sources, plugin_jar):
    outpath.makedirs()
    cls_files = list(outpath.walk_files(["class"]))
    sources = [v for v in sources if not is_up_to_date(v, cls_files)]
    src_by_typ = list(sources).equiv_classes(lambda a: a.endswith(".scala"))
    print("running compilers")
    try:
        if src_by_typ[False]:
            SubProc(["javac", "-cp", classpath, "-d", outpath] + src_by_typ[False]).start_wait()
    except:
        print("error running java")
        exit(1)
    try:
        if src_by_typ[True]:
            print("running scala on", src_by_typ[True])
            #os.enviorn["JAVA_OPTS"] = "-Xmx1024M -Xms256M -ea"
            SubProc(["fsc", "-cp", classpath + os.path.pathsep + outpath,
                "-d", outpath, "-Xplugin:" + str(plugin_jar)] + src_by_typ[True]).start_wait()
    except:
        print("error running scala")
        exit(1)
    print("done running compilers")

def main():
    classpath = Path("base/.scala_dependencies").read().split("\n")[0]
    classpath = classpath.replace("/skalch-base/", "/base/")
    classpath += os.path.pathsep + get_latest_jar("sketch-frontend") \
        + os.path.pathsep + Path("~/.m2/repository/org/scala-lang/scala-library/2.8.0.Beta1-RC8/scala-library-2.8.0.Beta1-RC8.jar")

    # === locate all source files ===
    files = Path("base").walk_files(["java", "scala"])
    files = [v for v in files if not "target" in v and "src" in v]
    main_sources = [v for v in files if "main" in v]
    other_sources = [v for v in files if not v in main_sources]

    # === locate the plugin ===
    plugin_jar = get_latest_jar("skalch-plugin")

    run_compile(classpath, Path("base/target/classes"), main_sources, plugin_jar)
    run_compile(classpath, Path("base/target/test-classes"), other_sources, plugin_jar)

if __name__ == "__main__":
    import optparse
    cmdopts = optparse.OptionParser()
    # cmdopts.add_option("--myflag", action="store_true", help="set my flag variable")
    noptions = len(cmdopts.option_list) - 1
    varargs = bool(main.__code__.co_flags & 0x04)
    required_args = main.__code__.co_argcount - noptions
    if varargs:
        cmdopts.usage = "%%prog [options] <<list %s>>" % (main.__code__.co_varnames[0])
    else:
        cmdopts.usage = "%prog [options] " + " ".join(
            v for v in main.__code__.co_varnames[:required_args])
    options, args = cmdopts.parse_args()
    if not varargs and required_args != len(args):
        cmdopts.error("%d arguments required." % (required_args))
    main(*args, **options.__dict__)
