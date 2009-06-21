package sketch.dyn;

import sketch.dyn.stats.ScStatOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.ui.ScUiOptions;
import sketch.util.CliParser;
import sketch.util.DebugOut;
import sketch.util.OptionResult;

public class BackendOptions {
    public static OptionResult synth_opts;
    public static OptionResult stat_opts;
    public static OptionResult ui_opts;

    /** add default lazy options to a parser */
    public static void add_opts(CliParser p) {
        DebugOut.assert_(synth_opts == null && (stat_opts == null),
                "adding command line options twice");
        synth_opts = (new ScSynthesisOptions()).parse(p);
        stat_opts = (new ScStatOptions()).parse(p);
        ui_opts = (new ScUiOptions()).parse(p);
    }

    /** initialize default options if the frontend didn't initialize them */
    public static void initialize_defaults() {
        if (synth_opts == null) {
            String[] no_args = {};
            DebugOut.print("please call BackendOptions.add_opts() "
                    + "in your frontend.");
            add_opts(new CliParser(no_args));
        }
    }
}
