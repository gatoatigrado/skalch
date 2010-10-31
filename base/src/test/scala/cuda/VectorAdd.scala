package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    def getb(b2 : IntArr @ scMemGlobal, off : Int) = b2(threadIdx.x + off)
    @scKernel def vectoradd(a : IntArr @ scInlineArray, alen : Int, b : IntArr)
    {
        val l2 : Int @scMemShared = alen;
        a(threadIdx.x) += getb(b, l2)
        __syncthreads()
    }

    @harness def sketchTest(x : Int) {
        vectoradd(new Array(100), 0, {val v = new Array[Int](100); v})
    }
}
