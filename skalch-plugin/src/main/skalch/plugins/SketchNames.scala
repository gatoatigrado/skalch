package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/**
 * Allocate names to SKETCH nodes.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
object SketchNames {
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
