package sketch.dyn;

import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;

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
    protected ScDynamicSketch ui_sketch;
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
        ui_sketch = f.apply();
        ssr = new ScStackSynthesis(sketches);
    }

    protected ScInputConf[] generate_inputs(ScDynamicSketch sketch) {
        ScTestGenerator tg = sketch.test_generator();
        tg.init(sketches[0].get_input_info());
        tg.tests();
        return tg.get_inputs();
    }

    public void synthesize() {
        ScInputConf[] inputs = generate_inputs(ui_sketch);
        // start various utilities
        ScUserInterface ui = ScUserInterfaceManager.start_ui(ssr, ui_sketch);
        ui.set_counterexamples(inputs);
        ScStats.stats.start_synthesis();
        // actual synthesize call
        ssr.synthesize(inputs, ui);
        // stop utilities
        ScStats.stats.stop_synthesis();
    }
}
