#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE - this file is mostly for examining compilation results, not for building the project

from __future__ import print_function
from collections import namedtuple
import optparse, os, re, subprocess, sys
import path_resolv
from path_resolv import Path

# setup classpath
os.environ["CLASSPATH"] = Path.pathjoin(os.environ["CLASSPATH"],
    path_resolv.resolve("lib/scala-library.jar") )
# reinit with updated classpath
path_resolv.resolvers[0] = path_resolv.EnvironPathResolver()

class Compiler(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.src_path = Path(self.src_path)
        self.out_path = Path(self.out_path)
        self.print = self.print or self.src_filter
        self.classpath = os.environ["CLASSPATH"]
        assert self.src_path.exists() and self.out_path.exists()

    def clean_dir(self):
        if self.clean:
            [self.out_path.subpath(f).remove_tree() for f in self.out_path.listdir()]

    def run(self, cmd, file_ext="txt"):
        if self.print:
            print(cmd)
        save_output = self.kwrite or self.vim
        pipe_output = { "stdout": subprocess.PIPE } if save_output else { }
        proc = subprocess.Popen(cmd, **pipe_output)
        text = proc.communicate()[0]
        if proc.returncode != 0:
            if not self.print: print("command", cmd)
            if text: print(text)
            raise Exception("subprocess returned error; code %d" %(proc.returncode))
        elif not save_output:
            return

        from tempfile import NamedTemporaryFile
        tf = NamedTemporaryFile(suffix="." + file_ext, delete=False)
        tf.write(text)
        tf.close()
        if self.kwrite:
            subprocess.Popen(["kwrite", tf.name], stderr=subprocess.PIPE)
        elif self.vim:
            subprocess.Popen(["vim", tf.name], stderr=subprocess.PIPE).wait()

    def compile_(self, src_file):
        assert src_file.exists()
        src_name = str(src_file)
        if self.src_java_names:
            src_name = Path(src_file.splitext()[0]).relpath(str(self.src_path)).replace("/", ".")
        with path_resolv.ExecuteIn(self.out_path):
            self.run([self.compiler] + [opt %(self.__dict__) for opt in self.option] +
                [src_name], file_ext="scala")

    def compile_all(self):
        for root, dirs, files in self.src_path.walk():
            filter_ = re.compile(self.src_filter) if self.src_filter else None
            for f in files:
                if not f.endswith(".scala"):
                    continue
                path = Path(root, f)
                if not filter_ or filter_.search(str(path)):
                    self.compile_(path)

    def run_app_(self):
        if self.run_app:
            self.kwrite = None
            self.vim = None
            # scala bug - `scala` doesn't work here
            subprocess.Popen(["java", "-classpath",
                "%s:%s" %(self.classpath, self.out_path), self.run_app] +
                [opt %(self.__dict__) for opt in self.run_option] ).wait()

def main(args):
    cmdopts = optparse.OptionParser(usage="%prog --options [files]",
        description="you might want to use the build.py wrapper for presets")
        #description="examples: ./custom_compile.py --dis_j --options= --kwr "
            #"src/test/TrivialTest2 to invoke the jvm, ./custom_compile.py "
            #"--options=-Xprint:jvm src/test/TrivialTest.scala to display the "
            #"jvm lowering phase")
    cmdopts.add_option("--src_filter", help="filter source files (when compiling all)")
    cmdopts.add_option("--src_path", default="src", help="path for src files")
    cmdopts.add_option("--out_path", default="bin", help="path for output files")
    cmdopts.add_option("--src_java_names", action="store_true",
        help="use full java names for source files (e.g. mypackage.ClassName)")
    cmdopts.add_option("--clean", action="store_true", help="clean the build directory")
    cmdopts.add_option("--compiler", default="scalac", help="compiler command to invoke")
    cmdopts.add_option("--option", action="append", default=[],
        help="options to pass to the compiler. example \"--opt -Xprint:jvm --opt -nowarn\".")
    cmdopts.add_option("--run_option", action="append", default=[],
        help="options to pass to java when/if running")
    cmdopts.add_option("--print", action="store_true", help="print command before executing")
    cmdopts.add_option("--run_app", help="run a class after compiling")
    cmdopts.add_option("--kwrite", action="store_true", help="launch kwrite")
    cmdopts.add_option("--vim", action="store_true", help="launch vim")
    options, args = cmdopts.parse_args(args)

    comp = Compiler(**options.__dict__)
    comp.clean_dir()
    if not args:
        comp.compile_all()
    else:
        for arg in args:
            comp.compile_(Path(arg))
    comp.run_app_()

if __name__ == "__main__":
    main(sys.argv[1:])
