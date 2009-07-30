package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

abstract class SketchNodeConnector[Identifier, From, To] {
    case class IdentifiedWaiting(lst : ListBuffer[From])
    case class IdentifiedDefined(to : To)

    val from_waiting = new HashMap[Identifier, AnyRef]

    /** abstract, override this */
    def connect(from : From, to : To)

    def connect_from(id : Identifier, from : From) = {
        from_waiting.get(id) match {
            case None => from_waiting.put(id,
                IdentifiedWaiting(ListBuffer(from)))
            case Some(IdentifiedWaiting(lst)) => lst += from
            case Some(IdentifiedDefined(to)) => connect(from, to)
        }
    }

    def connect_from(id : Identifier, to : To) = {
        from_waiting.get(id) match {
            case None => ()
            case Some(IdentifiedWaiting(lst)) => lst map (connect(_, to))
            case _ => DebugOut.assertFalse(
                "two sinks for identifier", id.toString, ";", to.toString)
        }
        from_waiting.put(id, IdentifiedDefined(to))
    }
}
