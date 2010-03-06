package sketch.dyn.main.old;

import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;

/**
 * Where everything begins.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesisMain extends ScSynthesisMainBase {
    protected final ScOldDynamicSketchCall[] sketches;
    protected final ScOldDynamicSketchCall ui_sketch;
    protected final ScSynthesis<?> synthesis_runtime;

    /**
     * Where everything begins.
     * @param f
     *            A scala function which will yield a new sketch
     * @param cmdopts
     *            Command options
     */
    public ScSynthesisMain(scala.Function0<ScOldDynamicSketch> f) {
        sketches = new ScOldDynamicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScOldDynamicSketchCall(f.apply());
        }
        ui_sketch = new ScOldDynamicSketchCall(f.apply());
        load_ui_sketch_info(ui_sketch);
        synthesis_runtime = get_synthesis_runtime(sketches);
    }

    protected ScSolvingInputConf[] generate_inputs(ScOldDynamicSketch sketch) {
        ScTestGenerator tg = sketch.test_generator();
        tg.init(ui_sketch.get_sketch().get_input_info());
        tg.tests();
        return tg.get_inputs();
    }

    public Object synthesize() {
        ScSolvingInputConf[] inputs = generate_inputs(ui_sketch.get_sketch());
        // start various utilities
        ScUserInterface ui =
                ScUserInterfaceManager.start_ui(be_opts, synthesis_runtime,
                        ui_sketch);
        init_stats(ui);
        ui.set_counterexamples(inputs);
        ui_sketch.counterexamples = ScFixedInputConf.from_inputs(inputs);
        for (ScOldDynamicSketchCall sketch : sketches) {
            sketch.counterexamples = ScFixedInputConf.from_inputs(inputs);
        }
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(ui);
        // stop utilities
        ScStatsMT.stats_singleton.showStatsWithUi();
        return synthesis_runtime.get_solution_tuple();
    }
}
