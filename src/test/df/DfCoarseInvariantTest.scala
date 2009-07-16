package test.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfCoarseInvariantSketch() extends AbstractDfSketch() {
    val swap_indices = (for (i <- 0 until 10) yield 0).toArray

    def df_main() {
        skdprint_loc("df_main()")
        skdprint(abbrev_str())
        var red_idx, white_idx, blue_idx = 0
        for (i <- (0 until num_buckets)) {
            if (i > 0) {
                // add relation to the registers
                val swap_idx = !!(i + 1)
                swap_indices(i) = swap_idx
                swap(i, swap_idx)
                skdprint(swap_idx + " " + abbrev_str())
            }
            synthAssertTerminal(isCorrect(i + 1))
        }
        skdprint(
            ("" /: swap_indices.view(0, num_buckets))(_ + ", " + _.toString)
            substring 2)
    }
}

object DfCoarseInvariantTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfCoarseInvariantSketch())
    }
}
