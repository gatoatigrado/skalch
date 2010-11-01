package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    def getb(b2 : IntArr @ scMemGlobal, off : Int) = {
        assert (b2(threadIdx.x + off) == 0)
        0 }

    @scKernel def vectoradd(a_ : IntArr @ scInlineArray, alen : Int, b : IntArr)
    {
        // NOTE -- SKETCH doesn't support writing to input arrays
        val a : IntArr @scMemGlobal = a_;

        a(threadIdx.x) += getb(b, 12)
        assert (a(threadIdx.x) == 0)
        a(threadIdx.x) += (?? : Int) * threadIdx.x;
        assert (a(threadIdx.x) == threadIdx.x + threadIdx.x)
    }

    @harness def sketchTest(x : Int) {
        vectoradd(new Array(100), 0, {val v = new Array[Int](100); v})
    }
}
