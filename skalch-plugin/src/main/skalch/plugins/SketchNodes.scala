package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/**
 * NOTE - this could be cleaned up using an object and
 * multiple definitions of functions for different types.
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
            case _ =>
                DebugOut.not_implemented("get_expr", node)
                null
        }
    }

    implicit def get_stmt(node : SketchNodeWrapper) : nodes.Statement = {
        node.node match {
            case stmt : nodes.Statement => stmt
            case _ =>
                DebugOut.not_implemented("get_stmt", node)
                null
        }
    }

    implicit def get_param(node : SketchNodeWrapper) : nodes.Parameter = {
        DebugOut.not_implemented("convert node to parameter\n", node.node)
        null
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
