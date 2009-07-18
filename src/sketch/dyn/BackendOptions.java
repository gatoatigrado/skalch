package sketch.dyn;

import sketch.dyn.ga.ScGaOptions;
import sketch.dyn.stats.ScStatsOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.ui.ScUiOptions;
import sketch.util.DebugOut;
import sketch.util.cli.CliOptionResult;
import sketch.util.cli.CliParser;

/**
 * all options used by the backend
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class BackendOptions {
    public static CliOptionResult synth_opts;
    public static ScUiOptions ui_opts;
    public static ScGaOptions ga_opts;
    public static ScStatsOptions stat_opts;

    /** add default lazy options to a parser */
    public static void add_opts(CliParser p) {
        if (synth_opts != null) {
            DebugOut.assertFalse("adding command line options twice");
        }
        synth_opts = (new ScSynthesisOptions()).parse(p);
        ui_opts = new ScUiOptions();
        ui_opts.parse(p);
        ga_opts = new ScGaOptions();
        ga_opts.parse(p);
        stat_opts = new ScStatsOptions();
        stat_opts.parse(p);
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

    public static void initialize_annotated() {
        ga_opts.set_values();
        stat_opts.set_values();
    }
}
