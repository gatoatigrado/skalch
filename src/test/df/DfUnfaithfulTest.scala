package test.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfUnfaithfulSketch(num_buckets : Int) extends AbstractDfSketch(num_buckets) {
    def df_main() {
        for (i <- (0 to !!(num_buckets))) {
            import java.lang.Integer
            swap(!!(buckets.length), !!(buckets.length))
        }
    }
}

object DfUnfaithfulTest {
    object TestOptions extends cli.CliOptionGroup {
        import java.lang.Integer
        add("--num_buckets", 10 : Integer, "number of buckets (each with one pebble)")
    }

    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfUnfaithfulSketch(
            opts.long_("num_buckets").intValue))
    }
}
