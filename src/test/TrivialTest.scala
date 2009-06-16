package test
import skalch.DynamicSketch
import sketch.util.DebugOut

object TrivialTest extends DynamicSketch {
    // this much is the same
    val hole0 = new Hole(untilv=3)
    val hole1 = new Hole(untilv=4)
    val hole2 = new Hole(untilv=6)
    val oracle0 = new OracleInput(untilv=10)

    def dysketch_main() = {
        val in0 : Int = next_int_input()
        assert(in0 + 1 == hole0() + in0) // regular scala asserts for now
        hole1()
        hole2()
        assert(hole1() == 2)
        assert(oracle0() == in0)
    }

    object ExhaustiveTestGenerator extends TestGenerator {
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set(x : Int, v : Boolean) {
            set_default_input(Array(x))
        }
        def tests() { for (ctr <- 0 until 10) test_case(new java.lang.Integer(ctr), new java.lang.Boolean(false)) }
    }

    def main(args : Array[String])  = {
        TrivialTest.synthesize_from_test(ExhaustiveTestGenerator)
    }
}
