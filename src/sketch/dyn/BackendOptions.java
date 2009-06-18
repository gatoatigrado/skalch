package sketch.dyn;

import sketch.dyn.stats.ScStatOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.util.CliParser;
import sketch.util.DebugOut;
import sketch.util.OptionResult;

public class BackendOptions {
    public static OptionResult synth_opts;
    public static OptionResult stat_opts;

    public static void add_opts(CliParser p) {
        DebugOut.assert_(synth_opts == null && (stat_opts == null),
                "adding command line options twice");
        synth_opts = (new ScSynthesisOptions()).parse(p);
        stat_opts = (new ScStatOptions()).parse(p);
    }
}
