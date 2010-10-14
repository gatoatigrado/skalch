package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

class VectorAdd() extends CudaKernel {
    type IntArr = Array[Int]

    def getb(b : IntArr, off : Int) = b(threadIdx.x + off)

    @scKernel def vectoradd(a : IntArr, alen : Int, b : IntArr) {
        a(threadIdx.x) += getb(b, alen)
    }
}
