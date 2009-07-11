package sketch.dyn;

import sketch.dyn.ga.ScGaOptions;
import sketch.dyn.stats.ScStatOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.ui.ScUiOptions;
import sketch.util.DebugOut;
import sketch.util.cli.CliParser;
import sketch.util.cli.CliOptionResult;

public class BackendOptions {
    public static CliOptionResult synth_opts;
    public static CliOptionResult stat_opts;
    public static CliOptionResult ui_opts;
    public static CliOptionResult ga_opts;

    /** add default lazy options to a parser */
    public static void add_opts(CliParser p) {
        if (synth_opts != null) {
            DebugOut.assertFalse("adding command line options twice");
        }
        synth_opts = (new ScSynthesisOptions()).parse(p);
        stat_opts = (new ScStatOptions()).parse(p);
        ui_opts = (new ScUiOptions()).parse(p);
        ga_opts = (new ScGaOptions()).parse(p);
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
