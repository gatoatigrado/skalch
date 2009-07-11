package sketch.dyn.synth;

import sketch.dyn.BackendOptions;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.ui.ScUserInterface;

/**
 * Base for stack and GA synthesis backends.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScSynthesis {
    // command line options
    public long nsolutions_to_find;
    public long debug_stop_after;
    public int max_num_random;

    public ScSynthesis() {
        // command line options
        nsolutions_to_find = BackendOptions.synth_opts.long_("num_solutions");
        debug_stop_after = BackendOptions.synth_opts.long_("debug_stop_after");
        max_num_random = (int) BackendOptions.ui_opts.long_("max_num_random");
    }

    public abstract boolean synthesize(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui);
}
