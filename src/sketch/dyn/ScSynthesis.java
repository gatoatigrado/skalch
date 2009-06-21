package sketch.dyn;

import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.util.DebugOut;
import sketch.util.Profiler;

/**
 * Where everything begins.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesis {
    protected int nthreads;
    protected ScDynamicSketch[] sketches;
    protected ScStackSynthesis ssr;

    /**
     * Where everything begins.
     * @param f
     *            A scala function which will yield a new sketch
     * @param cmdopts
     *            Command options
     */
    public ScSynthesis(scala.Function0<ScDynamicSketch> f) {
        // initialization
        BackendOptions.initialize_defaults();
        ScStats.initialize();
        nthreads = BackendOptions.synth_opts.int_("num_threads");

        // initialize ssr
        sketches = new ScDynamicSketch[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = f.apply();
        }
        ssr = new ScStackSynthesis(sketches);
    }

    public void synthesize() {
        ScTestGenerator tg = sketches[0].test_generator();
        tg.init(sketches[0].get_input_info());
        tg.tests();
        ScInputConf[] inputs = tg.get_inputs();
        if (BackendOptions.synth_opts.bool_("print_counterexamples")) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN,
                    "[user requested print]", "\n", true, text);
        }

        // start various utilities
        ScUserInterface ui = ScUserInterfaceManager.start_ui(ssr);
        Profiler.start_monitor();
        ScStats.stats.start_synthesis();

        // actual synthesize call
        ssr.synthesize(inputs, ui);

        // stop utilities
        ScStats.stats.stop_synthesis();
        Profiler.stop_monitor();
        ScStats.print_if_enabled();

        // don't do this upon completion, but the api is available
        // ScUiThread.stop_ui();
    }
}
