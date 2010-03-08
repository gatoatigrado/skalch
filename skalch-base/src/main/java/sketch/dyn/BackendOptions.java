package sketch.dyn;

import sketch.dyn.stats.ScStatsOptions;
import sketch.dyn.synth.ScSynthesisOptions;
import sketch.ui.ScUiOptions;
import sketch.util.DebugOut;
import sketch.util.cli.CliParser;
import sketch.util.sourcecode.ScSourceCache;
import ec.util.ThreadLocalMT;

/**
 * all options used by the backend
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class BackendOptions {
    public ScSynthesisOptions synthOpts;
    public ScUiOptions uiOpts;
    public ScStatsOptions statOpts;
    public static ThreadLocal<BackendOptions> backendOpts =
            new ThreadLocal<BackendOptions>();

    public BackendOptions(CliParser p) {
        synthOpts = new ScSynthesisOptions();
        synthOpts.parse(p);
        uiOpts = new ScUiOptions();
        uiOpts.parse(p);
        statOpts = new ScStatsOptions();
        statOpts.parse(p);
    }

    public static void addOpts(CliParser p) {
        backendOpts.set(new BackendOptions(p));
    }

    /** initialize default options if the frontend didn't initialize them */
    public static void initializeDefaults() {
        if (backendOpts.get() == null) {
            String[] noArgs = {};
            DebugOut.print("please call BackendOptions.addOpts() " + "in your frontend.");
            backendOpts.set(new BackendOptions(new CliParser(noArgs)));
        }
    }

    public void initializeAnnotated() {
        statOpts.set_values();
        DebugOut.no_bash_color = uiOpts.noBashColor;
        ScSourceCache.linesep = uiOpts.linesepRegex;
        ThreadLocalMT.disable_use_current_time_millis = synthOpts.noClockRand;
    }
}
