package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

import scala.tools.nsc
import nsc._

/**
 * Main map from Scala AST nodes to SKETCH AST nodes.
 * Bugs are most likely in the SketchTypes or SketchNames files
 * or in SKETCH lowering. This code is rather straightforward, since
 * I add a bunch of proxy nodes to the SKETCH AST.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class ScalaSketchNodeMap {
    val global : Global
    val ctx : nodes.FENode
    import global._

    val goto_connect : AutoNodeConnector[Symbol]
    val class_connect : AutoNodeConnector[Symbol]
    val class_fcn_connect : AutoNodeConnector[Symbol]

    import SketchNodes.{SketchNodeWrapper, SketchNodeList,
        get_expr, get_stmt, get_param, get_expr_arr,
        get_stmt_arr, get_param_arr, get_object_arr}

    def subtree(tree : Tree, next_info : ContextInfo = null) : SketchNodeWrapper
    def subarr(arr : List[Tree]) : SketchNodeList
    def gettype(tpe : Type) : nodes.Type
    def gettype(tree : Tree) : nodes.Type
    def getname(elt : Object, sym : Symbol) : String
    def getname(elt : Object) : String

    def unaryExpr(code : Int, target : nodes.Expression) : Object = {
        val sc = scalaPrimitives
        import nodes.{ExprUnary => sk}

        code match {
            case sc.ZNOT => new nodes.ExprUnary(ctx, sk.UNOP_NOT, target)
            case sc.NOT => new nodes.ExprUnary(ctx, sk.UNOP_BNOT, target)
            case sc.NEG => new nodes.ExprUnary(ctx, sk.UNOP_NEG, target)
        }
    }

    def binaryExpr(code : Int, left : nodes.Expression, right : nodes.Expression) : Object = {
        val sc = scalaPrimitives
        import nodes.{ExprBinary => sk}

        code match {
            case sc.ADD => new nodes.ExprBinary(ctx, sk.BINOP_ADD, left, right)
            case sc.SUB => new nodes.ExprBinary(ctx, sk.BINOP_SUB, left, right)
            case sc.MUL => new nodes.ExprBinary(ctx, sk.BINOP_MUL, left, right)
            case sc.DIV => new nodes.ExprBinary(ctx, sk.BINOP_DIV, left, right)
            case sc.MOD => new nodes.ExprBinary(ctx, sk.BINOP_MOD, left, right)

            case sc.EQ => new nodes.ExprBinary(ctx, sk.BINOP_EQ, left, right)
            case sc.NE => new nodes.ExprBinary(ctx, sk.BINOP_NEQ, left, right)
            case sc.LT => new nodes.ExprBinary(ctx, sk.BINOP_LT, left, right)
            case sc.LE => new nodes.ExprBinary(ctx, sk.BINOP_LE, left, right)
            case sc.GT => new nodes.ExprBinary(ctx, sk.BINOP_GT, left, right)
            case sc.GE => new nodes.ExprBinary(ctx, sk.BINOP_GE, left, right)

            case sc.AND => new nodes.ExprBinary(ctx, sk.BINOP_BAND, left, right)
            case sc.OR => new nodes.ExprBinary(ctx, sk.BINOP_BOR, left, right)
            case sc.XOR => new nodes.ExprBinary(ctx, sk.BINOP_BXOR, left, right)

            case sc.LSL => new nodes.ExprBinary(ctx, sk.BINOP_LSHIFT, left, right)
            case sc.ASR => new nodes.ExprBinary(ctx, sk.BINOP_RSHIFT, left, right)

            case _ => assertFalse("bad arithmetic op")
        }
    }

    def arrayExpr(code : Int, target : nodes.Expression, args : SketchNodeList) : Object = {
        val sc = scalaPrimitives

        if (sc.isArrayNew(code)) {
            not_implemented("array new")
        } else if (sc.isArrayLength(code)) {
            not_implemented("array length")
        } else if (sc.isArrayGet(code)) (args.list match {
            case Array(idx) => new nodes.ExprArrayRange(target, idx)
            case _ => not_implemented("array get with unexpected args", args)
        }) else if (sc.isArraySet(code)) (args.list match {
            case Array(idx, expr) => new nodes.StmtAssign(ctx,
                new nodes.ExprArrayRange(target, idx), expr)
            case _ => not_implemented("array set with unexpected args", args)
        }) else {
            assertFalse("no matching array opcode for code", code : Integer)
        }
    }

    /**
     * NOTE - main translation method.
     */
    def execute(tree : Tree, info : ContextInfo) : Object = {
        tree match {
            // much code directly copied from GenICode.scala (BSD license);
            // that file is a lot more complete though.
            case Apply(fcn @ Select(New(tpt), nme.CONSTRUCTOR), args) =>
                val ctor_sym = fcn.symbol

                import global.icodes._
                toTypeKind(tpt.tpe) match {
                    case arr : ARRAY => not_implemented("array constructor")
                    case ref_typ @ REFERENCE(cls_sym) =>
                        assert(ctor_sym.owner == cls_sym, "new sym owner not class")
                        val thisobj = class_connect.connect_from(cls_sym,
                            new exprs.ScalaExprClsNew(ctx))
                        thisobj.init_call = class_fcn_connect.connect_from(ctor_sym,
                            new exprs.ScalaClassFunctionCall(ctx, thisobj, subarr(args)))
                        thisobj
                    case _ => not_implemented("unknown new usage")
                }

            case Apply(fcn, args) =>
                val fcnsym = fcn.symbol
                if (fcnsym.isLabel) {
                    return goto_connect.connect_from(fcnsym, new misc.ScalaGotoCall(ctx))
                }

                // otherwise, it's of the form (object or class).(function name)
                val Select(target_tree, fcn_name) = fcn
                val target = subtree(target_tree)
                if (scalaPrimitives.isPrimitive(fcnsym)) {
                    val code = scalaPrimitives.getPrimitive(fcnsym, target_tree.tpe)
                    if (scalaPrimitives.isArrayOp(code)) {
                        arrayExpr(code, target, subarr(args))
                    } else (args match {
                        case Nil => unaryExpr(code, target)
                        case right :: Nil => binaryExpr(code, target, subtree(right))
                        case _ => assertFalse("bad argument list for arithmetic op", target,
                            code : java.lang.Integer, args)
                    })
                } else if (fcnsym.isStaticMember) {
                    not_implemented("static function call")
                    new nodes.ExprFunCall(ctx, getname(fcn), subarr(args))
                } else if (fcnsym.isClassConstructor) {
                    not_implemented("class constructor call")
                } else {
                    class_fcn_connect.connect_from(fcnsym,
                        new exprs.ScalaClassFunctionCall(ctx, target, subarr(args)))
                }

            case ArrayValue(elemtpt, elems) =>
                val unused = subtree(elemtpt)
                not_implemented("unused: elemtpt", unused)
                new nodes.ExprArrayInit(ctx, subarr(elems))

            case Assign(lhs, rhs) =>
                new nodes.StmtAssign(ctx, subtree(lhs), subtree(rhs))

            case Bind(name, body) =>
                new stmts.ScalaBindStmt(
                    ctx, gettype(body), getname(name), subtree(body))

            case Block(stmt_arr, expr) =>
                new stmts.ScalaBlock(ctx, subarr(stmt_arr), subtree(expr))

            case CaseDef(pat, guard, body) =>
                new stmts.ScalaCaseStmt(
                    ctx, subtree(pat), subtree(guard), subtree(body))

            case ClassDef(mods, name, tparams, impl) =>
                val next_info = new ContextInfo(info)
                next_info.curr_clazz = new typs.ScalaClass(
                    ctx, getname(name), subarr(tparams))
                DebugOut.print("class symbol", tree.symbol)
                class_connect.connect_to(tree.symbol, next_info.curr_clazz)
                subtree(impl, next_info)
                next_info.curr_clazz

            case DefDef(mods, name, tparams, vparamss, tpe, body) =>
                // info.curr_clazz
                val params = vparamss match {
                    case Nil => List[ValDef]()
                    case vparams :: Nil => vparams
                    case _ =>
                        assertFalse("unknown defdef params", vparamss)
                }
                // add the return node
                val body_stmt = (subtree(body).node match {
                    case stmt : nodes.Statement => stmt
                    case expr : nodes.Expression => new nodes.StmtReturn(ctx, expr)
                })
                class_fcn_connect.connect_to(tree.symbol,
                    new misc.ScalaClassFunction(ctx, nodes.Function.FUNC_PHASE,
                        tree.symbol.isStaticMember,
                        info.curr_clazz,
                        getname(name), gettype(tpe), subarr(params), body_stmt))

            case Ident(name) =>
                new nodes.ExprVar(ctx, getname(name))

            case If(cond, thenstmt, elsestmt) => new nodes.StmtIfThen(
                    ctx, subtree(cond), subtree(thenstmt), subtree(elsestmt))

            case LabelDef(name, params, rhs) =>
                goto_connect.connect_to(tree.symbol, new misc.ScalaGotoLabel(
                    ctx, getname(name), subarr(params), subtree(rhs)))

            case Literal(Constant(value)) => value match {
                    case v : Boolean => new nodes.ExprConstBoolean(ctx, v)
                    case v : Char => new nodes.ExprConstChar(ctx, v)
                    case v : Float => new nodes.ExprConstFloat(ctx, v)
                    case v : Int => new nodes.ExprConstInt(ctx, v)
                    case v : String => new nodes.ExprConstStr(ctx, v)
                    case () => new exprs.ScalaUnitExpression(ctx)
                    case _ => not_implemented("scala constant literal", value.toString)
                }

            case Match(selector, cases) =>
                new stmts.ScalaMatchStmt(
                    ctx, subtree(selector), subarr(cases))

            case New(tpt : Tree) =>
                new nodes.ExprNew(ctx, gettype(tpt))

            case PackageDef(pid, stats) =>
                DebugOut.print("stats", subarr(stats))
                new misc.ScalaPackageDef()

            case Return(expr) =>
                new nodes.StmtReturn(ctx, subtree(expr))

            case Select(qualifier, name) =>
                new nodes.ExprField(ctx, subtree(qualifier), getname(name))

            case Super(qual, mix) =>
                new vars.ScalaSuperRef(ctx, gettype(tree))

            case Template(parents, self, body) =>
                DebugOut.print("not visiting parents", parents)
                for (sketch_node <- subarr(body).list) sketch_node match {
                    case f : misc.ScalaClassFunction => ()
                    case _ =>
                        not_implemented("element", "'" + sketch_node + "'",
                            "in class body")
                }
                null

            // qual may reference an outer class.
            case This(qual) =>
                class_connect.connect_from(tree.symbol, new vars.ScalaThis(ctx))

            case Throw(expr) =>
                new stmts.ScalaThrow(ctx, subtree(expr))

            case Try(block, catches, finalizer) =>
                new stmts.ScalaTryCatchFinally(
                    ctx, subtree(block), subarr(catches), subtree(finalizer))

            case TypeApply(fcn, args) =>
                DebugOut.not_implemented("type apply", subtree(fcn), subarr(args))
                new exprs.ScalaTypeApply(ctx, null, null)

            case TypeTree() => gettype(tree)

            case Typed(expr, typ) =>
                new exprs.ScalaTypedExpression(
                    ctx, subtree(expr), gettype(typ))

            case ValDef(mods, name, typ, rhs) => new nodes.StmtVarDecl(
                ctx, gettype(typ), getname(name), subtree(rhs))

            case EmptyTree =>
                new exprs.ScalaEmptyExpression(ctx)

            case _ =>
                not_implemented("didn't match Scala node", tree)
        }
    }
}
