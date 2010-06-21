package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 until 32)

    def myfcn(z : Int) = (z * 5)
    def main(x : Int) {
        val myvalue = myfcn({ var x = 3; x += (?? : Int5); x; })
        assert(myvalue == 20)
    }
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
