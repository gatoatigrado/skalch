package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

import scala.tools.nsc
import nsc._

/**
 * NOTE - this could be cleaned up using an object and
 * multiple definitions of functions for different types.
 */
abstract class SketchTypes {
    val global : Global
    val ctx : nodes.FENode
    import global._

    def gettype(tpe : Type) : nodes.Type = {
        tpe.typeSymbol.fullNameString match {
            case "scala.Int" => nodes.TypePrimitive.int32type
            case "scala.Array" =>
                tpe.typeArgs match {
                    case Nil =>
                        DebugOut.assertFalse("array with no type args")
                        null
                    case t :: Nil => new nodes.TypeArray(gettype(t),
                        new proxy.ScalaUnknownArrayLength(ctx))
                    case lst =>
                        DebugOut.assertFalse("array with many args " + lst)
                        null
                }
            case "skalch.DynamicSketch$InputGenerator" =>
                new skproxy.ScalaInputGenType()
            case "skalch.DynamicSketch$HoleArray" =>
                new skproxy.ScalaHoleArrayType()
            case _ => DebugOut.not_implemented("gettype()",
                    tpe, tpe.typeSymbol.fullNameString)
                null
        }
    }
    def gettype(tree : Tree) : nodes.Type = gettype(tree.tpe)
}
