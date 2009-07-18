package sketch.dyn.synth;

import sketch.dyn.BackendOptions;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.ui.ScUserInterface;
import sketch.util.AsyncMTEvent;
import sketch.util.DebugOut;

/**
 * Base for stack and GA synthesis backends.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScSynthesis<LocalSynthType extends ScLocalSynthesis> {
    // command line options
    public long nsolutions_to_find;
    public long debug_stop_after;
    public int max_num_random;
    protected LocalSynthType[] local_synthesis;
    protected ScUserInterface ui;
    public ScExhaustedWaitHandler wait_handler;
    protected long nsolutions_found = 0;
    public AsyncMTEvent done_events = new AsyncMTEvent();

    public ScSynthesis() {
        // command line options
        nsolutions_to_find = BackendOptions.synth_opts.long_("num_solutions");
        debug_stop_after = BackendOptions.synth_opts.long_("debug_stop_after");
        max_num_random = BackendOptions.ui_opts.max_num_random_stacks;
    }

    public final void synthesize(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        this.ui = ui;
        wait_handler = new ScExhaustedWaitHandler(local_synthesis.length);
        done_events.reset();
        synthesize_inner(counterexamples, ui);
        for (ScLocalSynthesis local_synth : local_synthesis) {
            local_synth.thread_wait();
        }
        done_events.set_done();
    }

    protected abstract void synthesize_inner(
            ScSolvingInputConf[] counterexamples, ScUserInterface ui);

    protected final void increment_num_solutions() {
        nsolutions_found += 1;
        if (nsolutions_found == nsolutions_to_find) {
            DebugOut.print_mt("synthesis complete");
            wait_handler.set_synthesis_complete();
        }
    }
}
