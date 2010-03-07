package sketch.dyn.synth;

import java.util.concurrent.atomic.AtomicLong;

import sketch.dyn.BackendOptions;
import sketch.result.ScSynthesisResults;
import sketch.util.DebugOut;

/**
 * Base for stack and GA synthesis backends.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScSynthesis<LocalSynthType extends ScLocalSynthesis> {
    // command line options
    public long nsolutions_to_find;
    public long debug_stop_after;
    public int max_num_random;
    protected LocalSynthType[] local_synthesis;
    public ScExhaustedWaitHandler wait_handler;
    protected AtomicLong nsolutions_found;
    public final BackendOptions be_opts;
    protected ScSynthesisResults resultsStore;

    public ScSynthesis(BackendOptions be_opts) {
        nsolutions_found = new AtomicLong(0);
        this.be_opts = be_opts;
        // command line options
        nsolutions_to_find = be_opts.synth_opts.num_solutions;
        debug_stop_after = be_opts.synth_opts.debug_stop_after;
        max_num_random = be_opts.ui_opts.max_random_stacks;
    }

    public final void synthesize(ScSynthesisResults resultsStore) {
        this.resultsStore = resultsStore;
        wait_handler = new ScExhaustedWaitHandler(local_synthesis.length);
        synthesize_inner(resultsStore);
        for (ScLocalSynthesis local_synth : local_synthesis) {
            local_synth.thread_wait();
        }
    }

    protected abstract void synthesize_inner(ScSynthesisResults resultsStore);

    protected final void increment_num_solutions() {
        nsolutions_found.incrementAndGet();
        if (nsolutions_found.longValue() == nsolutions_to_find) {
            DebugOut.print_mt("synthesis complete");
            wait_handler.set_synthesis_complete();
        }
    }

    /** will be unpacked by fluffy scala match statement */
    public abstract Object get_solution_tuple();
}
