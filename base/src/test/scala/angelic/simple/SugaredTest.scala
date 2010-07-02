package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    def main() {
        val a = Array(1, 3, 4, 2)
        assert(a(?? : Int @ Range(0 to 2)) == 3)
//         assert(a.exists((2 ==):(Int => Boolean)))
    }
}
