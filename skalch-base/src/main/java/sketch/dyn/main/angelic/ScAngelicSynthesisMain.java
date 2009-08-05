package sketch.dyn.main.angelic;

import static sketch.dyn.BackendOptions.beopts;
import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.stack.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;

/**
 * where it all begins... for angelic sketches (see AngelicSketch.scala)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScAngelicSynthesisMain extends ScSynthesisMainBase {
    public final ScAngelicSketchCall ui_sketch;
    protected ScAngelicSketchCall[] sketches;
    protected ScSynthesis<?> synthesis_runtime;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        ui_sketch = new ScAngelicSketchCall(f.apply());
        if (beopts().ga_opts.enable) {
            synthesis_runtime = new ScGaSynthesis(sketches);
        } else {
            synthesis_runtime = new ScStackSynthesis(sketches);
        }
    }

    public Object synthesize() throws Exception {
        // start various utilities
        ScUserInterface ui =
                ScUserInterfaceManager.start_ui(synthesis_runtime, ui_sketch);
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(ui);
        // stop utilities
        ScStatsMT.stats_singleton.stop_synthesis();
        return synthesis_runtime.get_solution_tuple();
    }
}
