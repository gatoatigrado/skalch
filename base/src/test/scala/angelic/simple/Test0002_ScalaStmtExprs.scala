package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class TestScalaExprStmts() extends AngelicSketch {
    type Int7 = Int @ Range(-(1 << 6) until (1 << 6))

    /*
    def increment(arr : Array[Int]) {
        var i = 0
        while (i < arr.length) {
            arr(i) += 1
            i += 1
        }
    }
    */

    def main(x : Int) {
        var arr = Array(1, 2, 3)
        arr = new Array(4)
        // increment(arr)
        assert(arr(2) + (?? : Int7) == x)
    }
}
