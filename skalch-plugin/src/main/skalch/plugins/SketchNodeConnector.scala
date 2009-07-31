package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/**
 * Abstract class for resolving references to nodes not yet traversed,
 * e.g. forward jumps. Classes implementing this only
 * provide $connect : (from, to) -> unit$
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class SketchNodeConnector[Identifier, From, To] {
    case class IdentifiedWaiting(lst : ListBuffer[From])
    case class IdentifiedDefined(to : To)

    val map = new HashMap[Identifier, AnyRef]

    /** abstract, override this */
    def connect(from : From, to : To)

    def connect_from(id : Identifier, from : From) : From = {
        map.get(id) match {
            case None => map.put(id,
                IdentifiedWaiting(ListBuffer(from)))
            case Some(IdentifiedWaiting(lst)) => lst += from
            case Some(IdentifiedDefined(to)) => connect(from, to)
        }
        from
    }

    def connect_to(id : Identifier, to : To) : To = {
        map.get(id) match {
            case None => ()
            case Some(IdentifiedWaiting(lst)) => lst map (connect(_, to))
            case _ => assertFalse(
                "two sinks for identifier", id.toString, ";", to.toString)
        }
        map.put(id, IdentifiedDefined(to))
        to
    }

    def checkDone() {
        for (list <- map.values) list match {
            case IdentifiedWaiting(lst) =>
                assertFalse("sink for list", lst, "not found")
            case _ : IdentifiedDefined => ()
        }
    }
}
