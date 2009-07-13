package test.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfUnfaithfulSketch() extends AbstractDfSketch() {
    def df_main() {
        for (i <- (0 to !!(num_buckets))) {
            swap(!!(buckets.length), !!(buckets.length))
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
