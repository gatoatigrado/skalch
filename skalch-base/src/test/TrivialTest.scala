package test
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class TrivialSketch(val num_tests : Int) extends DynamicSketch {
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
        val v0 = oracle0()
        print("oracle " + v0 + ", in0 " + in0)
        v0 == in0
    }

    val test_generator = new TestGenerator {
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set(x : Int, v : Boolean) {
            put_default_input(x)
        }
        def tests() {
            for (ctr <- 0 until num_tests)
                test_case(new java.lang.Integer(ctr), new java.lang.Boolean(false))
        }
    }
}

object TrivialTest {
    object TestOptions extends cli.CliOptionGroup {
        add("--num_tests")
    }

    // good options
    // --array_length 3 --num_steps 5 --num_tests 10 --stat_enable
    def main(args : Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        val num_tests = TestOptions.parse(cmdopts).long_("num_tests").intValue
        skalch.synthesize(() => new TrivialSketch(num_tests))
    }
}
