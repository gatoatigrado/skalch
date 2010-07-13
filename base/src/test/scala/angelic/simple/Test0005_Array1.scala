package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

// TEST RULE FAILS

class TestArrays1() extends AngelicSketch {
    def main() {
        val a = Array(1, 3, 4, 2)
        assert(a(?? : Int @ Range(0 to 2)) == 2)
    }
}
