package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    def main(x : Int) {
        val a = ?? : Int @ Range(1 to 4)
        assert(a - 1 < 3)
        // val a = ?? : Array[Int] @ ArrayLen(5) @ Range(0 to 10)
        // assert(a(0) == 0)
        // tprint("array element zero" -> a(0),
            // "array element zero is zero" -> (a(0) == 0))
    }
}
