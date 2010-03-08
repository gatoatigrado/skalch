package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 to 32)
    def main(y : Int) {
        var x : Int5 = if (??) { -(?? : Int5) } else ??
        synthAssert(myfcn(x) + 4 == 3)
    }

    def myfcn(x : Int) = x + 1
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
