#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import optparse
import path_resolv
from path_resolv import Path

def check_file(f):
    text = f.read()
    if "\r" in text:
        raise Exception("FATAL - dos endlines in %s" %(f))
    for i, line in enumerate(text.split("\n")):
        def warn(text):
            print("%30s:%03d %s" %(f, i, text))
        if "\t" in line:
            warn("tabs present")
        # for now, ignore Eclipse blank comment lines
        if line.endswith(" ") and line.strip() != "*":
            warn("trailing whitespace")

def main(srcdir, file_extensions):
    assert type(file_extensions) == list
    for root, dirs, files in Path(srcdir).walk():
        for f in files:
            f = Path(root, f)
            if f.splitext()[-1][1:] in file_extensions:
                check_file(f)

if __name__ == "__main__":
    cmdopts = optparse.OptionParser(usage="%prog [options]")
    cmdopts.add_option("--srcdir", default=Path("."),
        help="source directory to look through")
    cmdopts.add_option("--file_extensions", default="java,scala,py,sh",
        help="comma-sepated list of file extensions")
    options, args = cmdopts.parse_args()
    options.file_extensions = options.file_extensions.split(",")
    main(**options.__dict__)
