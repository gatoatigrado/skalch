package skalch.cuda

import sketch.util.DebugOut

/**
 * Base class for a deterministic CUDA kernel
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class CudaKernel {
    class ParallelIndex() {
        val x : Int = 0
        val y : Int = 0
        val z : Int = 0
    }

    val threadIdx = new ParallelIndex()
    val blockDim = new ParallelIndex()
    val blockIdx = new ParallelIndex()
    val gridDim = new ParallelIndex()
}
