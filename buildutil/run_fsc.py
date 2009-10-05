#!/usr/bin/env python
# author: gatoatigrado (nicholas tung) [ntung at ntung]

from __future__ import division, print_function
from collections import namedtuple
import cPickle, optparse, os, shutil, subprocess
from path_resolv import Path, ExecuteIn

def run_quiet(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate()
    assert not proc.returncode is None
    return proc.returncode == 0

def run_noisy(cmd):
    print(" ".join(("'%s'" %(v) if " " in v else v) for v in cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    return proc.returncode == 0

def main(plugin_classpath, plugin_args_fname, plugin_clsdir, plugin_outname,
        base_classpath, base_args_fname, base_clsdir,
        test_classpath, test_args_fname, test_clsdir,
        compile_plugin=False, compile_base=False, compile_test=False,
        **kwargs):

    if not os.path.isdir(plugin_clsdir):
        os.makedirs(plugin_clsdir)
    plugin_cmd = ["fsc", "-classpath", plugin_classpath,
        "-deprecation", "@%s" %(plugin_args_fname)]
    if compile_plugin:
        if not run_noisy(plugin_cmd):
            print("FAILED, rerunning...")
            run_quiet(["fsc", "--reset"])
            if run_quiet(plugin_cmd):
                print("recompiling plugin succeeded")
            else:
                return
        [v.copy(plugin_clsdir) for v in Path("skalch-plugin/src/main/resources").files()]
        with ExecuteIn(Path(plugin_clsdir)):
            run_noisy(["jar", "c0f", plugin_outname] + Path(plugin_clsdir).listdir())

    if not os.path.isdir(base_clsdir):
        os.makedirs(base_clsdir)
    base_cmd = ["fsc", "-classpath", base_classpath,
        "-deprecation", "@%s" %(base_args_fname)]
    if compile_base:
        if not run_noisy(base_cmd):
            print("FAILED, rerunning...")
            run_quiet(["fsc", "--reset"])
            if run_quiet(base_cmd):
                print("recompiling base succeeded")
            else:
                return

    if not os.path.isdir(test_clsdir):
        os.makedirs(test_clsdir)
    test_cmd = ["fsc", "-classpath", test_classpath,
        "-deprecation", "@%s" %(test_args_fname)]
    if compile_test:
        if not run_noisy(test_cmd):
            print("FAILED, rerunning...")
            run_quiet(["fsc", "--reset"])
            if run_quiet(test_cmd):
                print("recompiling test succeeded")
            else:
                return

    #subprocess.Popen(["fsc", 

if __name__ == "__main__":
    cmdopts = optparse.OptionParser()
    cmdopts.add_option("--compile_plugin", action="store_true")
    cmdopts.add_option("--compile_base", action="store_true")
    cmdopts.add_option("--compile_test", action="store_true")
    options, args = cmdopts.parse_args()
    args = cPickle.load(open("buildutil/build_info.pickle"))
    args.update(options.__dict__)
    main(**args)
