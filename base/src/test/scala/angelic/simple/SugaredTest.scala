package angelic.simple
import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    class MyValue(val a : Int @ Range(3 to 444),
        val c : Int @ Range(List(3, 5)))

    def main() {
//         val elt = List(1, 2, 3)(!!)
//         val mv2 = SKF1(2)
        val v2 = !! : MyValue
        // TODO -- disallow this
//         skdprint((?? : Int @ Range(List(-1, 0))).toString)
        if (!!) synthAssert(v2.a == 4)
    }
}
