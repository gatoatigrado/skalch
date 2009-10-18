package skalch

import sketch.util.DebugOut

/**
 * New dynamic sketching base class. Most everything interesting is done
 * through the plugin; the Scala code here is only to maintani syntax/type
 * compatibility.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class AngelicSketch {
    abstract class _SketchConstruct
    object !! extends _SketchConstruct
    object ?? extends _SketchConstruct

    var stopOptimization : Boolean = false

    implicit def resolveBool(x : ??.type) : Boolean =
        { assert(false); false }
    implicit def resolveInt(x : ??.type) : Int =
        { assert(false); 0 }
    implicit def resolve[T <: AnyRef](x : ??.type) : T =
        { assert(false); null; }

//     implicit def resolveBool(x : !!.type) : boolean =
//         { assert(false); false }
    implicit def resolveInt(x : !!.type) : Int =
        { assert(false); if (stopOptimization) 272 else 8 }
    implicit def resolve[T <: Object](x : !!.type) : T =
        { assert(false); if (stopOptimization) null else "".asInstanceOf[T] }

//     def !!() : Boolean = { assert(false); false }
//     def !!() : Int = { assert(false); 0 }
//     def !![T <: AnyRef]() : T = { assert(false); null; }

//     def ??() : Boolean = { assert(false); false }
//     def ??() : Int = { assert(false); 0 }
//     def ??[T <: AnyRef]() : T = { assert(false); null; }

    def synthAssert(v : Boolean) { scala.Predef.assert(false); }
    def skdprint(x : => String) { assert(false); }
    def skdprint_loc(x : => String) { assert(false); }

    class Range[T](values : Seq[T]) extends StaticAnnotation

    // convenient type annotations
//     type SKF1 = FunctionType[Int][Int]
}
