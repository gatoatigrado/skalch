package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.util.gui.ScInputDialogs
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

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
    val ctx : nodes.FENode
    import global._

    val class_connect : AutoNodeConnector[Symbol]
    val external_refs : ListBuffer[String]

    val known_lib_types : Array[String] = Array(
        "scala.Function0")

    /** resolve a few built-in Scala types */
    def gettype(tpe : Type) : nodes.Type = {
        tpe.typeSymbol.fullNameString match {
            case "scala.Boolean" => nodes.TypePrimitive.booltype
            case "scala.Char" => nodes.TypePrimitive.int8type
            case "scala.Short" => nodes.TypePrimitive.int16type
            case "scala.Int" => nodes.TypePrimitive.int32type
            case "scala.Long" => nodes.TypePrimitive.int64type
            case "scala.Float" => nodes.TypePrimitive.floattype
            case "scala.Double" => nodes.TypePrimitive.doubletype
            case "scala.Array" =>
                tpe.typeArgs match {
                    case Nil => assertFalse("array with no type args")
                    case t :: Nil => new nodes.TypeArray(gettype(t),
                        new exprs.vars.ScalaUnknownArrayLength(ctx))
                    case lst => assertFalse("array with many args " + lst)
                }
            case "scala.runtime.BoxedUnit" =>
                new typs.ScalaUnitType()
            case "scala.Unit" | "java.lang.String" => nodes.TypePrimitive.strtype
            case "skalch.DynamicSketch$InputGenerator" =>
                new skconstr.ScalaInputGenType()
            case "skalch.DynamicSketch$HoleArray" =>
                new skconstr.ScalaHoleArrayType()
            case "skalch.AngelicSketch" =>
                new typs.ScalaAngelicSketchType()
            case typename => if (known_lib_types contains typename) {
                    external_refs += typename
                    new typs.ScalaExternalLibraryType(typename)
                } else {
                    class_connect.connect_from(tpe.typeSymbol,
                        new typs.ScalaUnknownType())
                }
        }
    }

    /** convert nodes to Expression nodes */
    implicit def get_expr(node : nodes.base.FEAnyNode) : nodes.Expression = {
        node match {
            case expr : nodes.Expression => expr
            case stmt : stmts.ScalaExprStmt =>
                new exprs.ScalaExprStmtWrapper(stmt)
            case assign : nodes.StmtAssign => new exprs.ScalaNonExpressionUnit(assign)
            case call : misc.ScalaGotoCall => new exprs.ScalaNonExpressionUnit(call)
            case lbl : misc.ScalaGotoLabel => new exprs.ScalaNonExpressionUnit(lbl)
            case ifthen : nodes.StmtIfThen => new exprs.ScalaExprIf(ifthen)
            case _ => not_implemented("get_expr", node)
        }
    }

    /** convert nodes to Statement nodes */
    implicit def get_stmt(node : nodes.base.FEAnyNode) : nodes.Statement = {
        node match {
            case stmt : nodes.Statement => stmt
            case unitexpr : exprs.ScalaUnitExpression => new nodes.StmtEmpty(unitexpr)
            case expr : nodes.Expression =>
                DebugOut.print("converting expression", expr, "to statement")
                new nodes.StmtExpr(expr)
            case _ => not_implemented("get_stmt", node)
        }
    }

    /** convert nodes to Type nodes */
    implicit def get_type_node(node : nodes.base.FEAnyNode) : nodes.Type = {
        node match {
            case typ : nodes.Type => typ
            case _ => not_implemented("convert node to type", node)
        }
    }

    implicit def get_param(node : nodes.base.FEAnyNode) : nodes.Parameter = {
        not_implemented("convert node to parameter\n", node)
    }

    def java_list[T](arr : Array[T]) : java.util.List[T] = {
        val result = new java.util.Vector[T]()
        for (v <- arr) { result.add(v) }
        java.util.Collections.unmodifiableList[T](result)
    }

    implicit def get_expr_arr(list : Array[nodes.base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_expr(elt)).toArray)
    }

    implicit def get_stmt_arr(list : Array[nodes.base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_stmt(elt)).toArray)
    }

    implicit def get_param_arr(list : Array[nodes.base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_param(elt)).toArray)
    }

    implicit def get_type_arr(list : Array[nodes.base.FEAnyNode]) = {
        java_list((for (elt <- list) yield get_type_node(elt)).toArray)
    }
}
