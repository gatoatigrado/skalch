package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 to 31)

    def main(x : Int) {
        assert((?? : Int5) * x == x + x + (?? : Int5))
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
