package sketch.dyn.main;

import static sketch.dyn.BackendOptions.beopts;
import sketch.dyn.BackendOptions;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.stack.ScStackSynthesis;
import ec.util.ThreadLocalMT;

public class ScSynthesisMainBase {
    protected int nthreads;

    public ScSynthesisMainBase() {
        BackendOptions.initialize_defaults();
        beopts().initialize_annotated();
        new ScStatsMT();
        nthreads = (int) beopts().synth_opts.long_("num_threads");
        ThreadLocalMT.disable_use_current_time_millis =
                beopts().synth_opts.bool_("no_clock_rand");
    }

    protected ScSynthesis<?> get_synthesis_runtime(
            ScDynamicSketchCall<?>[] sketches)
    {
        if (beopts().ga_opts.enable) {
            return new ScGaSynthesis(sketches);
        } else {
            return new ScStackSynthesis(sketches);
        }
    }
}
