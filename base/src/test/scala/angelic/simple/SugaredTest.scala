package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    def main() {
        val a = Array(1, 2, 4)
        assert(a(1) == 2)
//         assert(a.exists((2 ==):(Int => Boolean)))
    }
}
