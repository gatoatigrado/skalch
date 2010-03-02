package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 to 32)
//     class A(val a : Int @ Range(-3 to 1), val b : Boolean)
//     def b2(a : A) = a.a
    def main(y : Int) {
//         var y : Int = 0;
//         y = 3;
//         var z = 4;
//         var x : Int = if (??) { -(?? : Int) } else { val v = ?? : Int; v / 13 }
        val x = ?? : Int5
        synthAssert(x + 2 == 3)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
