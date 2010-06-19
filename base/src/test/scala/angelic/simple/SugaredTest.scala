package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    def main(xIn : Int) {
        var x = xIn;
        x = if (x >= 2 && x <= 10) 0 else x
        while (x > 10) {
            x -= (?? : Int @ Range(-32 to 310))
        }
        assert(x < 2)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
