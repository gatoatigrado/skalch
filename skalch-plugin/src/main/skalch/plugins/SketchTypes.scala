package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

import scala.tools.nsc
import nsc._

/**
 * Resolves a Scala type or type tree to a node
 * extending streamit.frontend.nodes.Type.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
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
                    case Nil => assertFalse("array with no type args")
                    case t :: Nil => new nodes.TypeArray(gettype(t),
                        new proxy.ScalaUnknownArrayLength(ctx))
                    case lst => assertFalse("array with many args " + lst)
                }
            case "skalch.DynamicSketch$InputGenerator" =>
                new skproxy.ScalaInputGenType()
            case "skalch.DynamicSketch$HoleArray" =>
                new skproxy.ScalaHoleArrayType()
            case _ => not_implemented("gettype()",
                    tpe, tpe.typeSymbol.fullNameString)
        }
    }
}
