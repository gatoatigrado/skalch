package test.df

import test.BitonicSortSketch
import skalch.DynamicSketch
import sketch.dyn.{BackendOptions, ScSynthesisBootstrapped}
import sketch.util._
import DebugOut.print

/**
 * Difficult test for developing the GA.
 * Currently takes backtracking search 9 659 860 iterations
 * in 37 seconds to find one solution
 * b fsc '/(Dynamic|Bootstrap)' run_app=test.df.BootstrapTestDifficult run_opt_list --ui_no_gui --sy_num_solutions 1
 * runs / sec: 261 225.56
 */
class BootstrapSketch() extends DynamicSketch {
    val synthesis_arr =
        (for (test_name <- BootstrapOptions.result.str_("tests").split(","))
            yield test_name match {
                case "bitonic-5" => new ScSynthesisBootstrapped(
                    () => new BitonicSortSketch(5, 10, 100))
        }).toArray

    def dysketch_main() : Boolean = {
        // TODO -- assert that all synthesis found solutions
        // TODO -- add cost based on number of trials
        // TODO -- use debug stop after
        for (synth <- synthesis_arr) {
            print("synthesis", synth)
        }
        true
    }

    val test_generator = NullTestGenerator
}

object BootstrapTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new BootstrapSketch())
    }
}

object BootstrapOptions extends cli.CliOptionGroup {
    var result : cli.CliOptionResult = null
    import java.lang.Integer
    add("--tests", "bitonic-5", "which tests to run and tune, comma separated " +
        "(options: bitonic-5)")
}
