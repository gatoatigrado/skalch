package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/// isolate rightmost zero bit
class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 to 32)

    /*
    def arrayToInt(arr : Array[Int]) = {
        var result = 0
        for (idx <- 0 until arr.length) {
            result += arr(idx) << idx
        }
        result
    }
    */

    def main() {
        assert((?? : Int5) < -2)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
