package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

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

    val goto_connect : SketchNodeConnector[
        Symbol, core.ScalaGotoCall, core.ScalaGotoLabel]
    val class_connect : SketchNodeConnector[
        Symbol, AnyRef, core.ScalaClass]

    import SketchNodes.{SketchNodeWrapper, SketchNodeList,
        get_expr, get_stmt, get_param, get_expr_arr,
        get_stmt_arr, get_param_arr, get_object_arr}

    def subtree(tree : Tree, next_info : ContextInfo = null) : SketchNodeWrapper
    def subarr(arr : List[Tree]) : SketchNodeList
    def gettype(tpe : Type) : nodes.Type
    def gettype(tree : Tree) : nodes.Type
    def getname(elt : Object, sym : Symbol) : String
    def getname(elt : Object) : String

    def execute(tree : Tree, info : ContextInfo) : Object = {
        tree match {
            // some code from GenICode.scala
            // that file is a lot more complete though.
            case Apply(fun, args) =>
                val sym = fun.symbol
                if (sym.isLabel) {
                    goto_connect.connect_from(sym, new core.ScalaGotoCall(ctx))
                } else if (scalaPrimitives.isPrimitive(sym)) {
                    // much taken from GenICode.scala
                    val Select(receiver, _) = fun
                    val code = scalaPrimitives.getPrimitive(sym, receiver.tpe)
                    val sketchCode = (code match {
                        case scalaPrimitives.ADD | scalaPrimitives.SUB |
                            scalaPrimitives.MUL => 0
                        case _ => 1
                    })
                    DebugOut.assertFalse("not implemented...")
                    null
                } else {
                    new nodes.ExprFunCall(ctx, getname(fun), subarr(args))
                }

            case ArrayValue(elemtpt, elems) =>
                val unused = subtree(elemtpt)
                DebugOut.not_implemented("unused: elemtpt", unused)
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
                next_info.curr_clazz = new core.ScalaClass(
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
                        DebugOut.assertFalse("unknown defdef params", vparamss)
                        null
                }
                // add the return node
                val body_stmt = (subtree(body).node match {
                    case stmt : nodes.Statement => stmt
                    case expr : nodes.Expression => new nodes.StmtReturn(ctx, expr)
                })
                new core.ScalaClassFunction(ctx, nodes.Function.FUNC_PHASE,
                    getname(name), gettype(tpe), subarr(params), body_stmt)

            case Ident(name) =>
                new nodes.ExprVar(ctx, getname(name))

            case If(cond, thenstmt, elsestmt) => new nodes.StmtIfThen(
                    ctx, subtree(cond), subtree(thenstmt), subtree(elsestmt))

            case LabelDef(name, params, rhs) =>
                goto_connect.connect_to(tree.symbol, new core.ScalaGotoLabel(
                    ctx, getname(name), subarr(params), subtree(rhs)))

            case Literal(value) =>
                DebugOut.not_implemented("scala constant literal", value)
                null
                // new vars.ScalaConstantLiteral()

            case Match(selector, cases) =>
                new stmts.ScalaMatchStmt(
                    ctx, subtree(selector), subarr(cases))

            case New(tpt : Tree) =>
                new nodes.ExprNew(ctx, gettype(tpt))

            case PackageDef(pid, stats) =>
                println("NOTE - new package...")
                DebugOut.print("stats", subarr(stats))
                new proxy.ScalaPackageDef()

            case Return(expr) =>
                new nodes.StmtReturn(ctx, subtree(expr))

            case Select(qualifier, name) =>
                new nodes.ExprField(ctx, subtree(qualifier), getname(name))

            case Super(qual, mix) =>
                new vars.ScalaSuperRef(ctx, gettype(tree))

            case Template(parents, self, body) =>
                DebugOut.print("not visiting parents", parents)
                for (sketch_node <- subarr(body).list) sketch_node match {
                    case f : core.ScalaClassFunction => ()
                    case _ =>
                        DebugOut.not_implemented("element", "'" + sketch_node + "'",
                            "in class body")
                        ()
                }
                null

            // qual may reference an outer class.
            case This(qual) =>
                val clazz = info.curr_clazz
                DebugOut.print(tree)
                DebugOut.print(tree.symbol)
                DebugOut.print(clazz)
                DebugOut.print(qual)
                DebugOut.not_implemented("this symbol")
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
                new proxy.ScalaTypedExpression(
                    ctx, subtree(expr), gettype(typ))

            case ValDef(mods, name, typ, rhs) => new nodes.StmtVarDecl(
                ctx, gettype(typ), getname(name), subtree(rhs))

            case EmptyTree =>
                new proxy.ScalaEmptyExpression(ctx)

            case _ =>
                DebugOut.print("not matched " + tree)
                null
        }
    }
}
