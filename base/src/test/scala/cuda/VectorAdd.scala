package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    @scKernel def vectoradd(a : IntArr, alen : Int) {
        val b : IntArr @scMemShared = null
    }

    @harness def sketchtest(x : Int) {
        vectoradd(new IntArr(8), 8)
    }
}
