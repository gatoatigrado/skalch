package skalch.plugins

import sketch.util.DebugOut
import streamit.frontend.nodes

object SketchNodes {
    class SketchNodeWrapper(val node : Object) {
        override def toString = "SketchNodeWrapper[" + node + "]"
    }
    class SketchNodeList(val list : Array[SketchNodeWrapper]) {
        override def toString = "SketchNodeList[" + list + "]"
    }

    // use views as hack around polymorphic function restriction
    implicit def get_expr(node : SketchNodeWrapper) : nodes.Expression = {
        node.node match {
            case expr : nodes.Expression => expr
            case block : nodes.scalaproxy.ScalaBlock =>
                new nodes.scalaproxy.ScalaExpressionBlock(block)
        }
    }

    implicit def get_stmt(node : SketchNodeWrapper) : nodes.Statement = {
        node.node match {
            case stmt : nodes.Statement => stmt
        }
    }

    implicit def get_param(node : SketchNodeWrapper) : nodes.Parameter = {
        DebugOut.assertFalse("convert node to parameter\n", node.node)
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
}
