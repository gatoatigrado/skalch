package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

// NOTE -- it should say 1, but the actual hole (after lowering) has value 0
class TestTprint() extends AngelicSketch {
    def main(x : Int) {
        val a = new Array[Int](5)
        assert(a(0) == 0)
        tprint("array element zero" -> a(0),
            "array element zero is zero" -> (a(0) == 0))
    }
}
