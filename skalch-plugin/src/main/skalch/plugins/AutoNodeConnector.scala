package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/**
 * SKETCH-node specific class for resolving edges across AST subtrees,
 * or useful back-edges. Traverses fields in $from$, find a public one
 * annotated with @AutoConnect("id")
 */
class AutoNodeConnector[Identifier](val acid : String) extends
    SketchNodeConnector[Identifier, AnyRef, AnyRef]
{
    def connect(from : AnyRef, to : AnyRef) {
        for (fld <- from.getClass.getFields
            if (fld.isAnnotationPresent(classOf[nodes.annot.AutoConnect]) &&
                (fld.getAnnotation(classOf[nodes.annot.AutoConnect]).value() == acid) ))
        {
            fld.set(from, to)
            return
        }
        DebugOut.print("from:", from)
        DebugOut.print("to:", to)
        DebugOut.assertFalse("AutoNodeConnector - couldn't find field assignable to")
    }
}
