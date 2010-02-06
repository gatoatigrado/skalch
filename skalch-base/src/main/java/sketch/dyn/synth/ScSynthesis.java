package sketch.dyn.synth;

import sketch.dyn.BackendOptions;
import sketch.ui.ScUserInterface;
import sketch.util.DebugOut;
import sketch.util.thread.AsyncMTEvent;

/**
 * Base for stack and GA synthesis backends.
 * 
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
    public ScUserInterface ui;
    public ScExhaustedWaitHandler wait_handler;
    protected long nsolutions_found = 0;
    public AsyncMTEvent done_events = new AsyncMTEvent();
    public final BackendOptions be_opts;

    public ScSynthesis(BackendOptions be_opts) {
        this.be_opts = be_opts;
        // command line options
        nsolutions_to_find = be_opts.synth_opts.num_solutions;
        debug_stop_after = be_opts.synth_opts.debug_stop_after;
        max_num_random = be_opts.ui_opts.max_random_stacks;
    }

    public final void synthesize(ScUserInterface ui) {
        this.ui = ui;
        wait_handler = new ScExhaustedWaitHandler(local_synthesis.length);
        done_events.reset();
        done_events.enqueue(ui, "synthesisFinished");
        synthesize_inner(ui);
        for (ScLocalSynthesis local_synth : local_synthesis) {
            local_synth.thread_wait();
        }
        done_events.set_done();
    }

    protected abstract void synthesize_inner(ScUserInterface ui);

    protected final void increment_num_solutions() {
        nsolutions_found += 1;
        if (nsolutions_found == nsolutions_to_find) {
            DebugOut.print_mt("synthesis complete");
            wait_handler.set_synthesis_complete();
        }
    }

    /** will be unpacked by fluffy scala match statement */
    public abstract Object get_solution_tuple();
}
