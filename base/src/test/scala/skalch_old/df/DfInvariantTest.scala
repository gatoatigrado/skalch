package skalch_old.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfInvariantSketch() extends AbstractDfSketch() {
    val swap_indices = (for (i <- 0 until 10) yield 0).toArray

    def df_main() {
        skdprint_loc("df_main()")
        skdprint(abbrev_str())
        var red_idx, white_idx, blue_idx = 0
        for (i <- (0 until num_buckets)) {
            if (i > 0) {
                // NOTE - from !!(num_buckets) to
                // !!(i + 1)
                // Failing sequence for !!(i) e.g.
                // r r r r r b
                // as last b can't be at end if using !!(i)

                // add relation to the registers
                val swap_idx = (buckets(i) match {
                    case Red() => red_idx
                    case White() => white_idx
                    case Blue() => blue_idx
                })

                swap_indices(i) = swap_idx
                swap(i, swap_idx)
                skdprint(swap_idx + ", " + red_idx + ", " + white_idx +
                    ", " + blue_idx + " " + abbrev_str())
            }
            red_idx += !!(2)
            white_idx += !!(2)
            blue_idx += !!(2)
            synthAssertTerminal(isCorrect(i + 1))
        }
        skdprint(
            ("" /: swap_indices.view(0, num_buckets))(_ + ", " + _.toString)
            substring 2)
    }
}

object DfInvariantTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfInvariantSketch())
    }
}
