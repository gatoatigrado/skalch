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
abstract class CudaKernel extends AngelicSketch {
    class TemplateType

    class ParallelIndex() {
        val x : Int = 0
        val y : Int = 0
        val z : Int = 0
    }

    val threadIdx = new ParallelIndex()
    val blockDim = new ParallelIndex()
    val blockIdx = new ParallelIndex()
    val gridDim = new ParallelIndex()

    case class IntPtr(val deref : Int) {
        @scSpecialFcn def atomicAdd (amnt : Int) = 0
        @scSpecialFcn def atomicExch(amnt : Int) = 0
        @scSpecialFcn def atomicCAS (amnt : Int) = 0
        @scSpecialFcn def atomicAnd (amnt : Int) = 0
        @scSpecialFcn def atomicXor (amnt : Int) = 0
        @scSpecialFcn def atomicOr  (amnt : Int) = 0
    }

    @scSpecialFcn def __syncthreads() = {}
}
