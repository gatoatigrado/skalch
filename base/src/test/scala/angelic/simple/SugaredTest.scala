package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int7 = Int @ Range(-(1 << 6) until (1 << 6))

    def main(x : Int) {
        var arr = Array(1, 2, 3)
        arr(2) = x + 3
        assert(arr(2) + (?? : Int7) == x)
    }
}
