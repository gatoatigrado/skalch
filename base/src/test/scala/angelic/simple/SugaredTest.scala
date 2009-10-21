package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    class MyValue(val a : Int @ Range(3 to 444),
        val c : Int @ Range(List(3, 5)))

    def main() {
        val v2 = !! : MyValue
        if (!!) synthAssert(v2.a == 4)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
