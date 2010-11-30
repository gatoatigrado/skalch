package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

object VectorAdd extends CudaKernel {
    type IntArr = Array[Int]

    @generator def upstep_off1 = (?? : Int @ Range(0 to 2))
    @generator def upstep_off2 = (?? : Int @ Range(1 to 3))
    @generator def adj = (?? : Int @ Range(-2 to 2))

    def upstep(a : IntArr @scMemGlobal, N : Int) {
        var delta = 1
        while (delta < N) {
            val a_i : Int = delta * (1 + 2 * threadIdx.x) - adj
            val b_i : Int = delta * (2 + 2 * threadIdx.x) - adj
            assert (a_i < b_i)
            if (b_i < N) {
                a(b_i) += a(a_i)
            }
            __syncthreads()
            delta *= 2
        }
    }

    @scKernel def vectoradd(a2 : IntArr, N : Int) { upstep(a2, N) }

    @harness def sketchtest(x : Int) {
        // output the result
        val arr2 = Array((?? : Int), (?? : Int), (?? : Int), (?? : Int),
                         (?? : Int), (?? : Int), (?? : Int), (?? : Int))
        val testArr = Array(1, 1, 1, 1,
                            0, 0, 0, 0)
        vectoradd(testArr, 4)
        assert (testArr(0) == 1)
        assert (testArr(1) == 2)

        var i = 0
        while (i < 8) {
            assert (arr2(i) == testArr(i))
            i += 1
        }
    }
}
