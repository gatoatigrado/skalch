#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess, sys

import build
import path_resolv

def remove_path(path):
    if path.exists():
        path.remove_tree()

def execute_cmd(cmd):
    print("executing", cmd)
    subprocess.Popen(cmd).wait()

def main():
    remove_path(path_resolv.Path("lib/sketchrewriter.jar"))
    remove_path(path_resolv.Path("bin/skalch/plugins"))

    build.main(sys.argv[1:] + ["no_default_opts", "--option=-classpath", "--option=%(classpath)s",
        "--option=-sourcepath", "--option=%(src_path)s", "/plugin", "print"])
    path_resolv.Path("src/skalch/plugins/scalac-plugin.xml").copy("bin")
    with path_resolv.ExecuteIn(path_resolv.Path("bin")):
        plugins = ["skalch/plugins/%s" %(f) for f in path_resolv.Path("skalch/plugins").listdir()]
        execute_cmd(["jar", "c0f", path_resolv.Path("../lib/sketchrewriter.jar"),
            "scalac-plugin.xml"] + plugins)
        execute_cmd(["fsc", "-reset"])


if __name__ == "__main__":
    main()

