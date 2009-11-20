#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2009 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

from __future__ import division, print_function
from collections import namedtuple
import re
import jinja2, jinja2.ext
from jinja2 import nodes, Environment, FileSystemLoader
from jinja2.ext import Extension

class SectionExtension(Extension):
    # a set of names that trigger the extension.
    tags = set(['section'])

    def __init__(self, environment):
        super(SectionExtension, self).__init__(environment)

        # add the defaults to the environment
        environment.extend(
            sections = { }
        )

    def parse(self, parser):
        lineno = parser.stream.next().lineno
        args = [nodes.Const(parser.parse_expression().name)]
        body = parser.parse_statements(['name:endsection'], drop_needle=True)
        return nodes.CallBlock(self.call_method('_set_section', args),
                               [], [], body).set_lineno(lineno)

    def _set_section(self, name, caller):
        """Helper callback."""
        assert not name in self.environment.sections, \
            "section '%s' redefined" %(name)
        result = self.environment.sections[name] = caller()
        return result

def generate_jinja2(path, base=None):
    base = base if base else path.parent()
    env = Environment(loader=FileSystemLoader(base),
        trim_blocks=True, extensions=[jinja2.ext.do, SectionExtension])
    unused = env.get_template(path.relpath(base)).render()
    return (env.sections, unused)
