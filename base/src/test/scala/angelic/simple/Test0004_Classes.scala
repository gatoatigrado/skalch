package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

// TEST RULE XMLOUT CONTAINS value="0"
class TestClasses() extends AngelicSketch {
    class Animal(val id : Int)

    def main() {
        val Louis = new Animal(1)
        val Constantino = new Animal(2)
        val chosen : Animal = if (??) Louis else Constantino
        synthAssert(chosen.id == 2)
    }
}
