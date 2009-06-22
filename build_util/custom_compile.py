#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# NOTE - this file is mostly for examining compilation results, not for building the project

from __future__ import print_function
from collections import namedtuple
import optparse, os, re, subprocess, sys
import path_resolv
from path_resolv import Path

# setup classpath
os.environ["CLASSPATH"] = Path.pathjoin(os.environ["CLASSPATH"],
    path_resolv.resolve("lib/scala-library.jar"),
    path_resolv.resolve("lib/scala-compiler.jar"))
# reinit with updated classpath
path_resolv.resolvers[0] = path_resolv.EnvironPathResolver()

class Compiler(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.src_path = Path(self.src_path)
        self.out_path = Path(self.out_path)
        self.print = self.print or self.src_filter
        self.classpath = os.environ["CLASSPATH"]
        self.src_extensions = self.src_extensions.split(",")
        assert self.src_path.exists() and self.out_path.exists()

    def clean_dir(self):
        if self.clean:
            [self.out_path.subpath(f).remove_tree() for f in self.out_path.listdir()]

    @staticmethod
    def print_cmd(cmd):
        def quote(v):
            v = re.sub("([\$\'\"])", r"\\\g<1>", v)
            return "\"%s\"" %(v) if (" " in v) else v
        print(" ".join([quote(v) for v in cmd]))

    def run(self, cmd, save_output=None, file_ext="txt"):
        if self.print:
            self.print_cmd(cmd)
        save_output = (self.kwrite or self.vim) if save_output is None else save_output
        pipe_output = { "stdout": subprocess.PIPE } if save_output else { }
        proc = subprocess.Popen(cmd, **pipe_output)
        text = proc.communicate()[0]
        if proc.returncode != 0:
            if not self.print: print("command", cmd)
            if text: print(text)
            sys.exit(1)
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

    def compile_(self, src_files, file_ext="scala"):
        assert all([src_file.exists() for src_file in src_files])
        if self.src_java_names:
            src_files = [Path(src_file.splitext()[0]).relpath(self.src_path).replace("/", ".")
                for src_file in src_files]
        with path_resolv.ExecuteIn(self.out_path):
            self.run([self.compiler] + [opt %(self.__dict__) for opt in self.option] +
                [str(v) for v in src_files], file_ext)

    def compile_all(self):
        src_files = set()
        for root, dirs, files in self.src_path.walk():
            filter_ = re.compile(self.src_filter) if self.src_filter else None
            for f in files:
                path = Path(root, f)
                if not path.extension() in self.src_extensions:
                    continue
                if not filter_ or filter_.search(str(path)):
                    src_files.add(path)
        if src_files:
            self.compile_(src_files)

    def run_app_(self):
        if self.run_app:
            # scala bug - `scala` doesn't work here
            self.run([self.javacmd] + [opt %(self.__dict__) for opt in self.java_option] + ["-classpath",
                "%s:%s" %(self.classpath, self.out_path), self.run_app] +
                [opt %(self.__dict__) for opt in self.run_option], save_output=False)

def get_cmdopts():
    cmdopts = optparse.OptionParser(usage="%prog --options [files]",
        description="you might want to use the build.py wrapper for presets")
    cmdopts.add_option("--src_filter", help="filter source files (when compiling all)")
    cmdopts.add_option("--src_path", default="src", help="path for src files")
    cmdopts.add_option("--out_path", default="bin", help="path for output files")
    cmdopts.add_option("--src_java_names", action="store_true",
        help="use full java names for source files (e.g. mypackage.ClassName)")
    cmdopts.add_option("--src_extensions", default="scala",
        help="source file extensions, e.g. \"scala,java\"")
    cmdopts.add_option("--clean", action="store_true", help="clean the build directory")
    cmdopts.add_option("--compiler", default="scalac", help="compiler command to invoke")
    cmdopts.add_option("--javacmd", default="java", help="java command to invoke")
    cmdopts.add_option("--java_option", action="append", default=[],
        help="options to pass to the run command (java)")
    cmdopts.add_option("--option", action="append", default=[],
        help="options to pass to the compiler. example \"--opt -Xprint:jvm --opt -nowarn\".")
    cmdopts.add_option("--run_option", action="append", default=[],
        help="options to pass to java when/if running")
    cmdopts.add_option("--print", action="store_true", help="print command before executing")
    cmdopts.add_option("--run_app", help="run a class after compiling")
    cmdopts.add_option("--kwrite", action="store_true", help="launch kwrite")
    cmdopts.add_option("--vim", action="store_true", help="launch vim")
    return cmdopts

def main(args):
    options, args = get_cmdopts().parse_args(args)

    comp = Compiler(**options.__dict__)
    comp.clean_dir()
    if not args:
        comp.compile_all()
    else:
        comp.compile_([Path(arg) for arg in args])
    comp.run_app_()

def get_options_help():
    return get_cmdopts().format_help()

if __name__ == "__main__":
    main(sys.argv[1:])
