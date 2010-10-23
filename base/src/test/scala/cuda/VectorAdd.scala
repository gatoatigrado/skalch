package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    def getb(b : IntArr, off : Int) = b(threadIdx.x + off)
    @scKernel def vectoradd(a : IntArr @ scInlineArray, alen : Int, b : IntArr) {
        val a = Array(1, 2, 3)
        a(threadIdx.x) += getb(b, alen)
        __syncthreads()
    }
}
