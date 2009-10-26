package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    type myint = Int @ Range(3 to 4)
    def annoyance(x : Boolean) = x
    def annoyance2(v1 : Int, v2 : Int, v3 : Int, z : Int, x : Boolean, y : Boolean) = x
    def main() {
        var x = 3
        synthAssert(annoyance(annoyance(annoyance2(
            x, // need to create a temporary reference for this variable
            3334, // but not this one.
            ?? : myint, // or this one, see BlockifySafe
            { x += 1; 0 },
            if (!!) true else !!,
            if (!!) true else !!))))
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
