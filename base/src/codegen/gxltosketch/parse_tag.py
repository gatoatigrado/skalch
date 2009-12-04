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

try:
    from gatoatigrado_lib import (ExecuteIn, Path, SubProc, get_singleton,
        list, memoize_file, pprint, set, sort)
except:
    raise ImportError("please install gatoatigrado's utility library from "
        "bitbucket.org/gatoatigrado/gatoatigrado_lib")

import spark

def sparktokencmp(self, other):
        return cmp(self.type, other)

def parsertoken(clsname, fields=""):
    return type(clsname, (namedtuple(clsname, "index text " + fields),),
        { "type": clsname.upper(), "__cmp__": sparktokencmp })

Word = parsertoken("Word")

class SyntacticToken(object):
    def __init__(self, index, text):
        self.index = index
        self.type = text
        assert(text == text.strip())

    def __cmp__(self, other):
        return cmp(self.type, other)

    def __str__(self):
        return " '%s' " % (self.type)
    __repr__ = __str__

class Scanner(spark.GenericScanner):
    start_index = 0

    def tokenize(self, input):
        self.rv = []
        spark.GenericScanner.tokenize(self, input)
        return self.rv

    def add(self, clsname, input):
        self.rv.append(clsname(self.start_index, input))
        self.start_index += len(input)

    def t_syntactic(self, input):
        r"[^\w\s,\(\)\{\}]+"
        self.add(SyntacticToken, input)

    def t_syntactic_singular(self, input):
        r"[,\(\)\{\}]"
        self.add(SyntacticToken, input)

    def t_word(self, input):
        r"\w+"
        self.add(Word, input)

    def t_whitespace(self, input):
        r"\s+"



# nodes
class ASTNode(object):
    def __repr__(self): return self.__class__.__name__

class TypeOnlyASTNode(ASTNode):
    def __init__(self, inner): self.inner = inner
    def __repr__(self):
        caps = "".join(v for v in self.__class__.__name__ if v.isupper())
        return " %s(%s) " % (caps, repr(self.inner))

class NameASTNode(ASTNode):
    def __init__(self, name): self.name = name
    def __repr__(self):
        caps = "".join(v for v in self.__class__.__name__ if v.isupper())
        return " %s('%s') " % (caps, self.name.text)

    def __str__(self):
        return self.name.text

class GxlArg(TypeOnlyASTNode): pass
class JavaArg(TypeOnlyASTNode): pass

class FcnName(NameASTNode): pass
class GxlSubtree(NameASTNode): pass
class GxlList(NameASTNode): pass
class JavaSubtree(NameASTNode): pass
class JavaImplicitArg(NameASTNode): pass



# lists
class AstList(ASTNode):
    def __init__(self, *argv):
        self.argv = argv

    def __repr__(self):
        args_str = ("\n" + "\n".join(repr(v) for v in self.argv)).replace("\n", "\n    ")
        return "AstList< %s >%s" % (", ".join(str(v.__class__.__name__) for v in self.argv), args_str)

class GxlArgs(AstList): pass
class JavaArgs(AstList): pass
class ConvertElts(AstList): pass

class ConvertElt(ASTNode):
    def __init__(self, javaname, java_args, gxlname, gxl_args):
        self.javaname = javaname
        self.java_args = java_args
        self.gxlname = gxlname
        self.gxl_args = gxl_args

    def __repr__(self):
        return " CE( %r (%r) -> %r (%r) ) " % (self.javaname, self.java_args, self.gxlname, self.gxl_args)

class Parser(spark.GenericASTBuilder):
    def __init__(self):
        spark.GenericASTBuilder.__init__(self, AST=ASTNode, start="ConvertElts")

    def p_scalatosketch(self, args):
        # N.B. -- anything ending with "s" is plural !!! meaning it will be
        # converted to an array (see below).
        r"""
        ConvertElts ::= ConvertElt
        ConvertElts ::= ConvertElt ConvertElt

        ConvertElt ::= FcnName ( GxlArgs ) -> FcnName ( JavaArgs )

        GxlArgs ::= GxlArg
        GxlArgs ::= GxlArg , GxlArgs
        GxlArg ::= GxlSubtree
        GxlArg ::= GxlList
        GxlSubtree ::= WORD
        GxlList ::= WORD []

        JavaArgs ::= JavaArg
        JavaArgs ::= JavaArg , JavaArgs
        JavaArg ::= JavaSubtree
        JavaArg ::= JavaImplicitArg
        JavaSubtree ::= WORD
        JavaImplicitArg ::= < WORD >

        FcnName ::= WORD
        """

    def nonterminal(self, nt, args):
        args = [v for v in args if not isinstance(v, SyntacticToken)]

        if nt.endswith("s"):
            result = []
            for arg in args:
                if arg.__class__.__name__ == nt and isinstance(arg, AstList):
                    result.extend(arg.argv)
                else:
                    result.append(arg)
            args = result

        if not nt in globals():
            from warnings import warn
            warn("nonterminal %s not found." % (nt))
            assert len(args) == 1, "args %r" % (args)
            return args[0]
        return globals()[nt](*args)

def parse_gxl_conversion(text):
    tokens = Scanner().tokenize(text)
    try:
        return Parser().parse(tokens)
    except spark.ParseError, e:
        print("tokenized successfully...")
        pprint(e.token)
