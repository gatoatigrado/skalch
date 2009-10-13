package skalch_old.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfUnfaithfulSketch() extends AbstractDfSketch() {
    def df_main() {
        skdprint_loc("df_main()")
        val num_steps = !!(num_buckets + 1)
        skAddCost(num_steps)
        for (i <- (0 until num_steps)) {
            swap(!!(num_buckets), !!(num_buckets))
            skdprint(abbrev_str())
        }
    }
}

object DfUnfaithfulTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfUnfaithfulSketch())
    }
}
