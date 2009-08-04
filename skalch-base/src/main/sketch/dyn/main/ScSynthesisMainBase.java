package sketch.dyn.main;

import static sketch.dyn.BackendOptions.beopts;
import sketch.dyn.BackendOptions;
import sketch.dyn.stats.ScStatsMT;
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
}
