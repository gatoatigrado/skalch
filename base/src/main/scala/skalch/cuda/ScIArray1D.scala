package skalch.cuda

import skalch.AngelicSketch
import skalch.cuda.annotations._
import sketch.util.DebugOut

/**
 * Base class for a deterministic CUDA kernel
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required,
 *          if you make changes, please consider contributing back!
 */

@scTemplateClass("T") abstract class ScIArray1D[T] {
    val length : Int = 0
    val values : Array[T] @scRetypeTemplateInner("T") @scInlineArray @scRawArray = null
    @scRetypeTemplate("T") def apply(idx : Int) = values(idx)
    def update(idx : Int, @scRetypeTemplate("T") value : T) { values(idx) = value; }
}

// @scTemplateInstanceType("Int") class ScIArray1D_Int extends ScIArray1D[Int]