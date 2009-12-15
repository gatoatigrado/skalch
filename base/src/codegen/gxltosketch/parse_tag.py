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

def sparktokencmp2(self, other):
    return 0 if (self.text == other) else cmp(self.type, other)

def parsertoken(clsname, fields="", matchText=False):
    return type(clsname, (namedtuple(clsname, "index text " + fields),),
        { "type": clsname.upper(), "__str__": lambda self: self.text,
            "__cmp__": sparktokencmp2 if matchText else sparktokencmp })

Word = parsertoken("Word", matchText=True)
String = parsertoken("String")

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
        r"[^\w_0-9\s\"]"
        self.add(SyntacticToken, input)

    def t_word(self, input):
        r"[\w\_0-9]+"
        self.add(Word, input)

    def t_whitespace(self, input):
        r"\s+"

    def t_string(self, input):
        r"\"(\\\"|[^\"])+\""
        self.add(String, input)



# nodes
class ASTNode(object):
    def __init__(self):
        self.cn = self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class NameASTNode(ASTNode):
    def __init__(self, name):
        self.name = name
        ASTNode.__init__(self)

    def __repr__(self):
        caps = "".join(v for v in self.__class__.__name__ if v.isupper())
        return " %s('%s') " % (caps, self.name.text)

    def __str__(self):
        return self.name.text

class FcnName(NameASTNode): pass
class GxlSubtree(NameASTNode): pass
class GxlAttribute(NameASTNode): pass
class GxlImplicitSubtree(NameASTNode): pass
class JavaSubtree(NameASTNode):
    def typename(self): return self.name.text
class JavaImplicitArg(NameASTNode):
    def argdecl(self, argname):
        if self.name == "ctx":
            return "FEContext %s = create_fe_context(node)" % (argname)
        elif self.name == "ctxnode":
            return "FENode %s = new DummyFENode(create_fe_context(node))" % (argname)
        else:
            assert self.name in ["null"]

    def inlinedecl(self):
        if self.name == "null":
            return "null"

class JavaEscapedArg(NameASTNode):
    def inlinedecl(self):
        return self.name.text[1:-1].replace(r"\"", "\"")

class JavaSubtreeList(NameASTNode):
    def __init__(self, list_string, name):
        NameASTNode.__init__(self, name)
    def typename(self): return "List<%s>" % (self.name.text)
class JavaSubtreeSingletonList(JavaSubtreeList): pass



class OptionalSpecAST(ASTNode):
    def __init__(self, *argv):
        [setattr(self, v, None) for v in self.SPEC[-1]]
        for subspec in self.SPEC:
            if len(subspec) == len(argv):
                [setattr(self, subspec[i], argv[i]) for i in range(len(subspec))]
                return ASTNode.__init__(self)
        assert False, "wrong number of args to %s" % (self.__class__.__name__)

class NewKw(OptionalSpecAST):
    SPEC = [ (), ("new",) ]
class GxlSubtreeOL(OptionalSpecAST):
    SPEC = [ ("name",), ("unused", "name") ]
class GxlSubtreeUL(OptionalSpecAST):
    SPEC = [ ("name",), ("unused", "name") ]
class JavaType(OptionalSpecAST):
    SPEC = [ (), ("name",) ]



# lists
class AstList(ASTNode):
    def __init__(self, *argv):
        self.argv = argv
        ASTNode.__init__(self)

    def __repr__(self):
        args_str = ("\n" + "\n".join(repr(v) for v in self.argv)).replace("\n", "\n    ")
        return "%s< %s >%s" % (self.cn, ", ".join(str(v.__class__.__name__) for v in self.argv), args_str)

    def __iter__(self):
        return self.argv.__iter__()

    def __getitem__(self, arg):
        return self.argv.__getitem__(arg)

class GxlArgs(AstList): pass
class JavaArgs(AstList): pass
class ConvertElts(AstList): pass
class GxlSubfieldArgs(AstList): pass

class ConvertElt(ASTNode):
    def __init__(self, gxlname, gxl_args, new_kw, javaname, java_args):
        self.gxlname = gxlname
        self.gxl_args = gxl_args
        self.new_kw = new_kw
        self.javaname = javaname# if not java_name_override else java_name_override
        self.java_args = java_args

    def java_type(self):
        return str(self.javaname)

    def return_rep(self):
        if isinstance(self.java_args, String):
            return str(self.java_args)[1:-1].replace(r"\"", "\"")

    def __repr__(self):
        return " CE( %r (%r) -> %r (%r) ) " % (self.gxlname, self.gxl_args,
            self.javaname, self.java_args)

class Parser(spark.GenericASTBuilder):
    def __init__(self):
        spark.GenericASTBuilder.__init__(self, AST=ASTNode, start="ConvertElts")

# NOTE -- spark doesn't work with this
# ConvertElt ::= FcnName ( GxlArgs ) - > NewKw FcnName ( JavaArgs ) : JavaType
# JavaType ::= NAME

    def p_scalatosketch(self, args):
        # N.B. -- anything ending with "s" is plural !!! meaning it will be
        # converted to an array (see below).
        r"""
        ConvertElts ::= ConvertElt
        ConvertElts ::= ConvertElt ConvertElts

        ConvertElt ::= FcnName ( GxlArgs ) - > NewKw FcnName ( JavaArgs )
        ConvertElt ::= FcnName ( GxlArgs ) - > NewKw FcnName STRING

        NewKw ::=
        NewKw ::= new

        GxlArgs ::=
        GxlArgs ::= GxlArg
        GxlArgs ::= GxlArg , GxlArgs
        GxlArg ::= GxlSubfieldArgs
        GxlSubfieldArgs ::= GxlArgInner : GxlSubfieldArgs
        GxlSubfieldArgs ::= GxlArgInner
        GxlSubfieldArgs ::= GxlSubfieldArgs . GxlAttribute
        GxlSubfieldArgs ::= . GxlAttribute

        GxlArgInner ::= GxlSubtree
        GxlArgInner ::= GxlImplicitSubtree
        GxlArgInner ::= GxlSubtreeOL
        GxlArgInner ::= GxlSubtreeUL
        GxlSubtree ::= WORD
        GxlImplicitSubtree ::= < WORD >
        GxlSubtreeOL ::= OL [ WORD ]
        GxlSubtreeUL ::= UL [ WORD ]
        GxlAttribute ::= WORD

        JavaArgs ::= JavaArg
        JavaArgs ::= JavaArg , JavaArgs
        JavaArg ::= JavaSubtree
        JavaArg ::= JavaSubtreeList
        JavaArg ::= JavaSubtreeSingletonList
        JavaArg ::= JavaImplicitArg
        JavaArg ::= JavaEscapedArg
        JavaSubtree ::= WORD
        JavaSubtreeList ::= List [ WORD ]
        JavaSubtreeSingletonList ::= SingletonList [ WORD ]
        JavaImplicitArg ::= < WORD >
        JavaEscapedArg ::= STRING

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

        if nt in ["GxlArgInner", "GxlArg", "JavaArg"]:
            return get_singleton(args)
        if not nt in globals():
            from warnings import warn
            warn("nonterminal %s not found." % (nt))
            assert len(args) == 1, "Please provide a nonterminal class " \
                "for %s\nargs %r" % (nt, args)
            return args[0]
        return globals()[nt](*args)

def parse_gxl_conversion(text):
    tokens = Scanner().tokenize(text)
#    try:
    return Parser().parse(tokens)
#    except spark.ParseError, e:
#        print("tokenized successfully; parser error")
#        print(tokens)
#        print(e)
#        a = e.token.index
#        print("code before: %s" % (text[max(0, a - 10):(a + 1)]))
#        print("code after: %s" % (text[(a + 1):min(len(text), a + 10)]))
#        raise
