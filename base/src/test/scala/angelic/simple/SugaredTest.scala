package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    class A(val a : Int @ Range(-1 to 1), val b : Boolean)
    def main() {
        var x : A = ??
        assert(x.b)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
