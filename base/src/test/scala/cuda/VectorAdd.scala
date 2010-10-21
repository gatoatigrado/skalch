package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    val off2 = 0
    def getb(b : IntArr, off : Int) = b(threadIdx.x + off + off2)

    @scKernel def vectoradd(a : IntArr @ scInlineArray, alen : Int, b : IntArr) {
        a(threadIdx.x) += getb(b, alen)
        __syncthreads()
    }
}
