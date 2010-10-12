package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

class VectorAdd() extends CudaKernel {
    type IntArr = Array[Int]

    //--------------------------------------------------
    // @scTemplateClass("T") class ScIArray1D[T] {
    //     val length : Int = 0
    //     val values : Array[T] @scRetype(classOf[Array[T]]) @scRetypeTemplateInner("T") @scInlineArray @scRawArray = null
    //     @scRetypeTemplate("T") def apply(idx : Int) = values(idx)
    //     def update(idx : Int, @scRetypeTemplate("T") value : T) { values(idx) = value; }
    // }
    // @scTemplateInstanceType(classOf[Int]) class ScIArray1D_Int extends ScIArray1D[Int]
    //-------------------------------------------------- 

    def getb(b : IntArr, off : Int) = b(threadIdx.x + off)

    @scKernel def vectoradd(a : IntArr, alen : Int, b : IntArr) {
        a(threadIdx.x) += getb(b, alen)
    }
}
