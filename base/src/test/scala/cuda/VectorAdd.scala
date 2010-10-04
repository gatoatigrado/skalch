package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

class VectorAdd() extends CudaKernel {
    // type IntArr = ScIArray1DInt

    class VLArray[T] {
        val length : Int = 0
        val values : Array[T] @scInlineArray @scRawArray = null
        def apply(idx : Int) = values(idx)
        def update(idx : Int, value : T) { values(idx) = value; }
    }

    def getb(b : VLArray[Int], off : Int) = b(threadIdx.x + off)

    @scKernel def vectoradd(a : VLArray[Int], b : VLArray[Int]) {
        a(threadIdx.x) += getb(b, a.length)
    }
}
