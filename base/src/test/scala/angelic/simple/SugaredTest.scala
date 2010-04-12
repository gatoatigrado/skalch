package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class MySketch() extends AngelicSketch {
    type Int5 = Int @ Range(-32 to 32)

    class Animal(val id : Int)
    // class Cat(id : Int, val striped : Boolean) extends Animal(id)

    def main(y : Int) {
        val Louis = new Animal(1)
        val Constantino = new Animal(2)
        val chosen : Animal = if (??) Louis else Constantino
        synthAssert(chosen.id == 2)
        // synthAssert(myfcn(chosen.id) == 2)
    }

     // def myfcn(x : Int) = x + 1
}

@SkalchIgnoreClass
class IrrelevantClass {
    def myirrelevantfunction() {
        println("hello")
    }
}
