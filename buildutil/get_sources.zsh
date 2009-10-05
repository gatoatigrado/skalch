#!/usr/bin/env zsh

mvn -e clean install -Dmaven.scala.displayCmd=true | \
    grep -E "^\[INFO\] cmd\: " | python -c 'import cPickle, re, shutil, sys
from os.path import sep, pathsep, realpath

def parse_cmd(text):
    cmd, file = re.split(r"@([^\s ]+)", text[13:])[:2]
    classpath = cmd.strip().split()[2]
    classpath = pathsep.join(v for v in classpath.split(pathsep) if not "scala-compiler" in v)
    return classpath, file

def get_outdir(args):
    assert type(args) == list
    return args[args.index("-d") + 1]

def main():
    plugin_classpath, fname = parse_cmd(sys.stdin.readline())
    plugin_clsdir = get_outdir(open(fname).read().split("\n"))
    plugin_args_fname = "buildutil/plugin.build-args"
    shutil.copy(fname, plugin_args_fname)
    plugin_outname = "%s%s%s" %(realpath("buildutil"), sep, "sketchrewriter.jar")

    base_classpath, base_fname = parse_cmd(sys.stdin.readline())
    base_args = open(base_fname).read().split("\n")
    base_clsdir = get_outdir(base_args)
    base_args[0] = "-Xplugin:%s" %(plugin_outname)
    base_args_fname = "buildutil/base.build-args"
    open(base_args_fname, "w").write("\n".join(base_args))
    del base_args

    test_classpath, test_fname = parse_cmd(sys.stdin.readline())
    test_args = open(test_fname).read().split("\n")
    test_clsdir = get_outdir(test_args)
    test_args[0] = "-Xplugin:%s" %(plugin_outname)
    test_args_fname = "buildutil/test.build-args"
    open(test_args_fname, "w").write("\n".join(test_args))
    del test_args

    cPickle.dump(locals(), open("buildutil/build_info.pickle", "w"))

main()
remaining = sys.stdin.read().strip()
assert not remaining, "more compile stages?\n" + remaining
'
