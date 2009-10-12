#!/usr/bin/env python
# author: gatoatigrado (nicholas tung) [ntung at ntung]

from __future__ import division, print_function
from collections import namedtuple
import cPickle, optparse, os, shutil, subprocess, sys
from path_resolv import Path, ExecuteIn

def run_quiet(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate()
    assert not proc.returncode is None
    return proc.returncode == 0

def strcmd(cmd):
    return " ".join(("'%s'" %(v) if " " in v else v) for v in cmd)

def run_noisy(cmd):
    print(strcmd(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    return proc.returncode == 0

def main(plugin_classpath, plugin_args_fname, plugin_clsdir, plugin_outname,
        base_classpath, base_args_fname, base_clsdir,
        test_classpath, test_args_fname, test_clsdir,
        compile_plugin=False, compile_base=False, compile_test=False,
        **kwargs):

    plugin_clsdir = Path(plugin_clsdir)
    plugin_clsdir.rmtree()
    plugin_clsdir.makedirs()
    plugin_cmd = ["fsc", "-classpath", plugin_classpath,
        "-deprecation", "@%s" %(plugin_args_fname)]
    if compile_plugin:
        if os.path.isfile(plugin_outname):
            Path(plugin_outname).unlink()
        if not run_noisy(plugin_cmd):
            return False
        [v.copyinto(plugin_clsdir) for v in Path("skalch-plugin/src/main/resources").files()]
        with ExecuteIn(plugin_clsdir):
            run_noisy(["jar", "c0f", plugin_outname] + plugin_clsdir.listdir())

    if not os.path.isdir(base_clsdir):
        os.makedirs(base_clsdir)
    os.environ["JAVA_OPTS"] = strcmd(["-cp", base_classpath])
    base_cmd = ["fsc", "-deprecation", "@%s" %(base_args_fname)]
    if compile_base:
        if not run_noisy(base_cmd):
            return False

    if not os.path.isdir(test_clsdir):
        os.makedirs(test_clsdir)
    test_cmd = ["fsc", "-classpath", test_classpath,
        "-deprecation", "@%s" %(test_args_fname)]
    if compile_test:
        if not run_noisy(test_cmd):
            return False

    return True

if __name__ == "__main__":
    cmdopts = optparse.OptionParser()
    cmdopts.add_option("--compile_plugin", action="store_true")
    cmdopts.add_option("--compile_base", action="store_true")
    cmdopts.add_option("--compile_test", action="store_true")
    options, args = cmdopts.parse_args()
    args = cPickle.load(open("buildutil/build_info.pickle"))
    args.update(options.__dict__)
    sys.exit(1 if not main(**args) else 0)
