package skalch_old.simple

import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.dyn.constructs.{ctrls, inputs}
import sketch.util.cli
import sketch.util.DebugOut._ // assertFalse, etc.

class SugaredSketch() extends DynamicSketch {

    def dysketch_main() = {
        synthAssertTerminal(??(List("a", "b", "c")) == "c")
        ??(100) == 63
    }

    val test_generator = new TestGenerator {
        def set(x : Int, v : Boolean) {
            put_default_input(x)
        }

        def tests() {
            for (ctr <- 0 until 4)
                test_case(new java.lang.Integer(ctr), new java.lang.Boolean(false))
        }
    }
}

object SugaredTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new SugaredSketch()) match {
            case null => assertFalse("This sketch is solvable! Values: 2, 63")
            case (ctrl_values : ctrls.ScCtrlConf, oracle_values : inputs.ScInputConf) =>
                print("got control values", ctrl_values.getValueArray().toString)
                assert(ctrl_values.getValueArray() deepEquals Array(2, 63))
            case other => assertFalse("unknown solution for sketch", other)
        }
    }
}
