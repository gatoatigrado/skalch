package sketch.dyn.main;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.not_implemented;
import sketch.dyn.BackendOptions;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.stack.ScStackSynthesis;

public class ScSynthesisMainBase {
    protected int nthreads;

    public ScSynthesisMainBase() {
        BackendOptions.initialize_defaults();
        beopts().initialize_annotated();
        nthreads = beopts().synth_opts.num_threads;
        new ScStatsMT();
    }

    protected ScSynthesis<?> get_synthesis_runtime(
            ScDynamicSketchCall<?>[] sketches)
    {
        if (beopts().synth_opts.solver.isGa) {
            return new ScGaSynthesis(sketches);
        } else if (beopts().synth_opts.solver.isStack) {
            return new ScStackSynthesis(sketches);
        } else {
            not_implemented("ScSynthesisMainBase -- create unknown solver",
                    beopts().synth_opts.solver);
            return null;
        }
    }
}
