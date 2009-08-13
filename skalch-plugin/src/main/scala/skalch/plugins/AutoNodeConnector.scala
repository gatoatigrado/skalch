package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.compiler.ast.{base, core, scala => scast}

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
            if (fld.isAnnotationPresent(classOf[scast.annot.AutoConnect])) )
        {
            val annot = fld.getAnnotation(classOf[scast.annot.AutoConnect])
            if (annot.value() == acid) {
                fld.set(from, to)
                if (!annot.onConnect.isEmpty) {
                    from.getClass.getMethod(annot.onConnect).invoke(from)
                }
            }
            return
        }
        DebugOut.print("from:", from)
        DebugOut.print("to:", to)
        assertFalse("AutoNodeConnector - couldn't find [public] field assignable to.")
    }
}
