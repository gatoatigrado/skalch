package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.util.gui.ScInputDialogs
import sketch.compiler.ast.{base, core, scala => scast}

import scala.tools.nsc
import nsc._

/**
 * Resolves a Scala type or type tree to a node
 * extending streamit.frontend.nodes.Type.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class SketchTypes {
    val global : Global
    val ctx : core.FENode
    import global._

    val class_connect : AutoNodeConnector[Symbol]
    val external_refs : ListBuffer[String]

    val known_lib_types : Array[String] = Array(
        "scala.Function0")

    /** resolve a few built-in Scala types */
    def gettype(tpe : Type) : core.typs.Type = {
        tpe.typeSymbol.fullNameString match {
            case "scala.Boolean" => core.typs.TypePrimitive.booltype
            case "scala.Char" => core.typs.TypePrimitive.int8type
            case "scala.Short" => core.typs.TypePrimitive.int16type
            case "scala.Int" => core.typs.TypePrimitive.int32type
            case "scala.Long" => core.typs.TypePrimitive.int64type
            case "scala.Float" => core.typs.TypePrimitive.floattype
            case "scala.Double" => core.typs.TypePrimitive.doubletype
            case "scala.Array" =>
                tpe.typeArgs match {
                    case Nil => assertFalse("array with no type args")
                    case t :: Nil => new core.typs.TypeArray(gettype(t),
                        new scast.exprs.vars.ScalaUnknownArrayLength(ctx))
                    case lst => assertFalse("array with many args " + lst)
                }
            case "scala.runtime.BoxedUnit" =>
                new scast.typs.ScalaUnitType()
            case "scala.Unit" | "java.lang.String" => core.typs.TypePrimitive.strtype
            case "skalch.DynamicSketch$InputGenerator" =>
                new scast.skconstr.ScalaInputGenType()
            case "skalch.DynamicSketch$HoleArray" =>
                new scast.skconstr.ScalaHoleArrayType()
            case "skalch.AngelicSketch" =>
                new scast.typs.ScalaAngelicSketchType()
            case typename => if (known_lib_types contains typename) {
                    external_refs += typename
                    new scast.typs.ScalaExternalLibraryType(typename)
                } else {
                    class_connect.connect_from(tpe.typeSymbol,
                        new scast.typs.ScalaUnknownType())
                }
        }
    }

    /** convert nodes to Expression nodes */
    implicit def get_expr(node : base.FEAnyNode) : core.exprs.Expression = {
        node match {
            case expr : core.exprs.Expression => expr
            case stmt : scast.stmts.ScalaExprStmt =>
                new scast.exprs.ScalaExprStmtWrapper(stmt)
            case assign : core.stmts.StmtAssign => new scast.exprs.ScalaNonExpressionUnit(assign)
            case call : scast.misc.ScalaGotoCall => new scast.exprs.ScalaNonExpressionUnit(call)
            case lbl : scast.misc.ScalaGotoLabel => new scast.exprs.ScalaNonExpressionUnit(lbl)
            case ifthen : core.stmts.StmtIfThen => new scast.exprs.ScalaExprIf(ifthen)
            case _ => not_implemented("get_expr", node)
        }
    }

    /** convert nodes to Statement nodes */
    implicit def get_stmt(node : base.FEAnyNode) : core.stmts.Statement = {
        node match {
            case stmt : core.stmts.Statement => stmt
            case unitexpr : scast.exprs.ScalaUnitExpression => new core.stmts.StmtEmpty(unitexpr)
            case expr : core.exprs.Expression =>
                DebugOut.print("converting expression", expr, "to statement")
                new core.stmts.StmtExpr(expr)
            case _ => not_implemented("get_stmt", node)
        }
    }

    /** convert nodes to Type nodes */
    implicit def get_type_node(node : base.FEAnyNode) : core.typs.Type = {
        node match {
            case typ : core.typs.Type => typ
            case _ => not_implemented("convert node to type", node)
        }
    }

    implicit def get_param(node : base.FEAnyNode) : core.Parameter = {
        not_implemented("convert node to parameter\n", node)
    }

    def java_list[T](arr : Array[T]) : java.util.List[T] = {
        val result = new java.util.Vector[T]()
        for (v <- arr) { result.add(v) }
        java.util.Collections.unmodifiableList[T](result)
    }

    implicit def get_expr_arr(list : Array[base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_expr(elt)).toArray)
    }

    implicit def get_stmt_arr(list : Array[base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_stmt(elt)).toArray)
    }

    implicit def get_param_arr(list : Array[base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_param(elt)).toArray)
    }

    implicit def get_type_arr(list : Array[base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_type_node(elt)).toArray)
    }
}
