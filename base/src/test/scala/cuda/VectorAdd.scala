package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

class VectorAdd() extends CudaKernel {
    @cudakernel def vectoradd(a : Array[Int], b : Array[Int]) {
        a(threadIdx.x) += b(threadIdx.x + a.length)
    }
}
