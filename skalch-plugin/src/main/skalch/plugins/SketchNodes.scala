package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes

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
            case block : nodes.scalaproxy.ScalaBlock =>
                new nodes.scalaproxy.ScalaExpressionBlock(block)
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

    case class LogicalName(val name : String,
        val symbolName : String,
        val typ : String, // could be "type", "term", ...
        val pedanticNameFcn : (String => String),
        val collisionNameFcn : ((Int, Int) => String))
    {
        var unique_name : String = null
        var verbosity = 0

        def setUniqueName(names : HashMap[String, ListBuffer[LogicalName]]) {
            val others = names.get(name)
            val (verbosity, conflicts) = (if (!others.isDefined) {
                (0, ListBuffer[LogicalName](this))
            } else {
                ((others.get.head.verbosity + 1), (others.get += this))
            })
            set_conflicts_verbosity(conflicts, verbosity)
            resolve_conflicts(conflicts)
        }

        def sketchName(pedantic : Boolean) : String =
            if (pedantic) { pedanticNameFcn(unique_name) } else { unique_name }

        def similarTo(other : LogicalName) : Float = {
            (if (name == other.name) 1.f else 0.f) +
            (if (other.name contains name) 0.5f else 0.f) +
            (if (typ == other.typ) 0.1f else 0.f) +
            (if (symbolName == other.symbolName) 2.f else 0.f)
        }
    }

    def set_conflicts_verbosity(lst : ListBuffer[LogicalName], verbosity : Int) = {
        for ((elt, i) <- lst.zipWithIndex) {
            elt.unique_name = elt.collisionNameFcn(i, verbosity)
            elt.verbosity = verbosity
        }
    }

    def resolve_conflicts(lst : ListBuffer[LogicalName]) {
        val names = HashSet[LogicalName]()
        for (elt <- lst) {
            if (!names.add(elt)) {
                // conflicts
                set_conflicts_verbosity(lst, elt.verbosity + 1)
                resolve_conflicts(lst)
            }
        }
    }

    class NameStringFactory(val pedantic : Boolean) {
        val names = ListBuffer[LogicalName]()
        val conflicts = HashMap[String, ListBuffer[LogicalName]]()

        def get_name(query : LogicalName) : String = {
            if (query == null) {
                DebugOut.assertFalse()
            }

            def add() : LogicalName = {
                query.setUniqueName(conflicts)
                names += query
                query
            }

            ((for (name <- names)
                yield (name.similarTo(query), name)
            ).toList match {
                case Nil => add()
                case lst =>
                    val best = (lst.head /: lst.tail)(
                        (x, y) => if (x._1 >= y._1) x else y)
                    if (best._1 >= 1.f) best._2 else add()
            }).sketchName(pedantic)
        }
    }
}
