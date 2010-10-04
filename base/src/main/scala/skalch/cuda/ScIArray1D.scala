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

class ScIArray1D[T] {
    val length : Int = 0
    val values : Array[T] @scInlineArray @scRawArray = null
    def apply(idx : Int) = values(idx)
    def update(idx : Int, value : T) { values(idx) = value; }
}