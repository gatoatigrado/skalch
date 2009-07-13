package test.df

class DfUnfaithfulSketch(num_buckets : Int) extends AbstractDfSketch(num_buckets) {
}

object DfUnfaithfulTest {
    object TestOptions extends cli.CliOptionGroup {
        import java.lang.Integer
        add("--num_buckets", 10 : Integer, "length of list")
    }

    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfUnfaithfulSketch(
            opts.long_("num_buckets").intValue)
    }
}
