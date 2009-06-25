package test

import ec.util.ThreadLocalMT
import skalch.DynamicSketch

import sketch.ui.sourcecode.ScSourceLocation
import sketch.dyn.ctrls.ScCtrlSourceInfo
import sketch.dyn.BackendOptions
import sketch.util._

class TrivialDynamicUntilv(val max_n : Int, val num_tests : Int)
    extends DynamicSketch
{
    val in_length = new InputGenerator(untilv=(1 << 30))
    val oracle = new OracleInput(untilv=max_n)

    def dysketch_main() = {
        val n = in_length()
        val x = oracle()
        dynamicUntilvAssert(x <= n)
        x == n
    }

    val test_generator = new TestGenerator {
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set() { put_input(in_length, mt.get().nextInt(max_n)) }
        def tests() { for (i <- 0 until num_tests) test_case() }
    }

    // generate this code with the plugin
    {
        // val filename = "/home/gatoatigrado/sandbox/eclipse/skalch/src/test/TrivialDynamicUntilv.scala"
        // val line_num = (line : Int) => new ScSourceLocation(filename, line)
        // addHoleSourceInfo(new ScCtrlSourceInfo(swap_first_idx, line_num(41)))
        // addHoleSourceInfo(new ScCtrlSourceInfo(swap_second_idx, line_num(42)))
    }
}

object TrivialDynamicUntilvTest {
    object TestOptions extends CliOptGroup {
        add("--max_value", 100 : java.lang.Integer, "value for !! to guess")
        add("--num_tests", 10 : java.lang.Integer,
            "number of randomly generated input sequences to test")
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new TrivialDynamicUntilv(
            opts.long_("max_value").intValue, opts.long_("num_tests").intValue))
    }
}
