package sketch.dyn;

import sketch.dyn.stats.ScStatsOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.dyn.synth.ga.ScGaOptions;
import sketch.ui.ScUiOptions;
import sketch.util.DebugOut;
import sketch.util.cli.CliParser;
import sketch.util.sourcecode.ScSourceCache;
import ec.util.ThreadLocalMT;

/**
 * all options used by the backend
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class BackendOptions {
    public ScSynthesisOptions synth_opts;
    public ScUiOptions ui_opts;
    public ScGaOptions ga_opts;
    public ScStatsOptions stat_opts;
    public static ThreadLocal<BackendOptions> backend_opts =
            new ThreadLocal<BackendOptions>();

    public BackendOptions(CliParser p) {
        synth_opts = new ScSynthesisOptions();
        synth_opts.parse(p);
        ui_opts = new ScUiOptions();
        ui_opts.parse(p);
        ga_opts = new ScGaOptions();
        ga_opts.parse(p);
        stat_opts = new ScStatsOptions();
        stat_opts.parse(p);
    }

    public static void add_opts(CliParser p) {
        backend_opts.set(new BackendOptions(p));
    }

    /** initialize default options if the frontend didn't initialize them */
    public static void initialize_defaults() {
        if (backend_opts.get() == null) {
            String[] no_args = {};
            DebugOut.print("please call BackendOptions.add_opts() "
                    + "in your frontend.");
            backend_opts.set(new BackendOptions(new CliParser(no_args)));
        }
    }

    public void initialize_annotated() {
        ga_opts.set_values();
        stat_opts.set_values();
        DebugOut.no_bash_color = ui_opts.no_bash_color;
        ScSourceCache.linesep = ui_opts.linesep_regex;
        ThreadLocalMT.disable_use_current_time_millis =
                synth_opts.no_clock_rand;
    }
}
