package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
//     class A(val a : Int @ Range(-3 to 1), val b : Boolean)
//     def b2(a : A) = a.a
    def main(y : Int) {
        var x : Int = if (??) { -?? } else ??
        synthAssert(x + 4 == 3)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
