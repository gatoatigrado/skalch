package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    def annoyance(x : Boolean) = x
    def main() {
        synthAssert(annoyance(annoyance(annoyance(if (!!) true else !!))))
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
