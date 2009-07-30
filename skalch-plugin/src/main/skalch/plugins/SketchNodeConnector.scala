package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

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
            case _ => DebugOut.assertFalse(
                "two sinks for identifier", id.toString, ";", to.toString)
        }
        map.put(id, IdentifiedDefined(to))
        to
    }

    def checkDone() {
        for (list <- map.values) list match {
            case IdentifiedWaiting(lst) =>
                DebugOut.assertFalse("sink for list", lst, "not found")
            case _ : IdentifiedDefined => ()
        }
    }
}
