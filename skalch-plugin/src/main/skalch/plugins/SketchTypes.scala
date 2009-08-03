package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.util.gui.ScInputDialogs
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

    val known_lib_types : Array[String] = Array(
        "scala.Function0")

    def gettype(tpe : Type) : nodes.Type = {
        tpe.typeSymbol.fullNameString match {
            case "scala.Char" => nodes.TypePrimitive.int8type
            case "scala.Short" => nodes.TypePrimitive.int16type
            case "scala.Int" => nodes.TypePrimitive.int32type
            case "scala.Long" => nodes.TypePrimitive.int64type
            case "scala.Float" => nodes.TypePrimitive.floattype
            case "scala.Double" => nodes.TypePrimitive.doubletype
            case "scala.Array" =>
                tpe.typeArgs match {
                    case Nil => assertFalse("array with no type args")
                    case t :: Nil => new nodes.TypeArray(gettype(t),
                        new vars.ScalaUnknownArrayLength(ctx))
                    case lst => assertFalse("array with many args " + lst)
                }
            case "skalch.DynamicSketch$InputGenerator" =>
                new skconstr.ScalaInputGenType()
            case "skalch.DynamicSketch$HoleArray" =>
                new skconstr.ScalaHoleArrayType()
            case typename =>
                if ((known_lib_types contains typename) ||
                    ScInputDialogs.yesno("continue with unknown class " + typename, this))
                {
                    new typs.ScalaExternalLibraryType(typename)
                } else {
                    not_implemented("unknown type", tpe, typename)
                }
        }
    }
}
