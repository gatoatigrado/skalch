package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

// NOTE -- it should say 1, but the actual hole (after lowering) has value 0
// TEST RULE XMLOUT CONTAINS value="0"
class TestWhileLoops() extends AngelicSketch {
    type Int7 = Int @ Range(-(1 << 6) until (1 << 6))
    type Natural32 = Int @ Range(1 to 32)

    def max(x : Int, y : Int) = if (x < y) y else x

    def main(x : Int) {
        var i = 0
        while (i <= x) {
            i += max(x, 2) - (?? : Natural32);
        }
        assert(i > x);
    }
}
