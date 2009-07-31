package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

import scala.tools.nsc
import nsc._

/**
 * Uses Scala views to convert expressions to statements and vice versa.
 * NOTE - this could be cleaned up using an object and
 * multiple definitions of functions for different types.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
object SketchNodes {
    class SketchNodeWrapper(val node : Object) {
        override def toString = "SketchNodeWrapper[" + node + node.getClass + "]"
    }
    class SketchNodeList(val list : Array[SketchNodeWrapper]) {
        override def toString = {
            ("SketchNodeList[" /: list)(_ + _ + ", ") + " ]"
        }
    }

    // use views as hack around polymorphic function restriction
    implicit def get_expr(node : SketchNodeWrapper) : nodes.Expression = {
        node.node match {
            case expr : nodes.Expression => expr
            case stmt : exprs.ScalaExprStmt =>
                new exprs.ScalaExprStmtWrapper(stmt)
            case _ => not_implemented("get_expr", node)
        }
    }

    implicit def get_stmt(node : SketchNodeWrapper) : nodes.Statement = {
        node.node match {
            case stmt : nodes.Statement => stmt
            case _ => not_implemented("get_stmt", node)
        }
    }

    implicit def get_param(node : SketchNodeWrapper) : nodes.Parameter = {
        not_implemented("convert node to parameter\n", node.node)
    }

    def java_list[T](arr : Array[T]) : java.util.List[T] = {
        val result = new java.util.Vector[T]()
        for (v <- arr) { result.add(v) }
        java.util.Collections.unmodifiableList[T](result)
    }

    implicit def get_expr_arr(list : SketchNodeList) = {
        java_list((for (elt <- list.list) yield get_expr(elt)).toArray)
    }

    implicit def get_stmt_arr(list : SketchNodeList) = {
        java_list((for (elt <- list.list) yield get_stmt(elt)).toArray)
    }

    implicit def get_param_arr(list : SketchNodeList) = {
        java_list((for (elt <- list.list) yield get_param(elt)).toArray)
    }

    implicit def get_object_arr(list : SketchNodeList) = {
        java_list[AnyRef]((for (elt <- list.list) yield elt).toArray)
    }
}
