package sketch.dyn.synth;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import sketch.dyn.BackendOptions;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScSynthCtrlConf;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.prefix.ScDefaultPrefix;
import sketch.dyn.prefix.ScPrefixSearchManager;
import sketch.ui.ScUserInterface;
import sketch.util.DebugOut;
import sketch.util.MTReachabilityCheck;

/**
 * cloned Lexin's implementation, then modified for
 * <ul>
 * <li>support for oracle values (!!)</li>
 * <li>backtracking / exploration in multiple sparse regions.</li>
 * </ul>
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStackSynthesis {
    protected ScLocalStackSynthesis[] local_synthesis;
    protected ScSynthCtrlConf ctrls;
    protected ScInputConf oracle_inputs;
    protected ScUserInterface ui;
    protected int nsolutions_found = 0;
    // command line options
    protected int nsolutions_to_find;
    protected int debug_stop_after;
    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> search_manager;
    public ExhaustedWaitHandler wait_handler;
    public MTReachabilityCheck reachability_check;

    public ScStackSynthesis(ScDynamicSketch[] sketches) {
        // initialize backends
        local_synthesis = new ScLocalStackSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] =
                    new ScLocalStackSynthesis(sketches[a], this, a);
        }
        ScDefaultPrefix prefix = new ScDefaultPrefix();
        ScStack stack =
                new ScStack(sketches[0].get_hole_info(), sketches[0]
                        .get_oracle_input_list(), prefix);
        // shared classes to synchronize / manage search
        search_manager = new ScPrefixSearchManager<ScStack>(stack, prefix);
        // command line options
        nsolutions_to_find = BackendOptions.synth_opts.int_("num_solutions");
        debug_stop_after = BackendOptions.synth_opts.int_("debug_stop_after");
    }

    public boolean synthesize(ScInputConf[] counterexamples, ScUserInterface ui)
    {
        this.ui = ui;
        wait_handler = new ExhaustedWaitHandler();
        reachability_check = new MTReachabilityCheck();
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            ui.addStackSynthesis(local_synth);
            local_synth.run(counterexamples);
        }
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            local_synth.thread_wait();
        }
        return wait_handler.synthesis_complete.get();
    }

    public synchronized void add_solution(ScStack stack) {
        ui.addSolution(stack);
        nsolutions_found += 1;
        if (nsolutions_found == nsolutions_to_find) {
            DebugOut.print_mt("synthesis complete");
            wait_handler.set_synthesis_complete();
        }
    }

    public class ExhaustedWaitHandler {
        protected AtomicInteger n_exhausted = new AtomicInteger(0);
        protected Semaphore wait = new Semaphore(0);
        public AtomicBoolean synthesis_complete = new AtomicBoolean(false);

        public void wait_exhausted() {
            if (n_exhausted.incrementAndGet() >= local_synthesis.length) {
                DebugOut.print_mt("all exhausted, exiting");
                set_synthesis_complete();
                n_exhausted.addAndGet(-(local_synthesis.length));
                return;
            }
            DebugOut.print_mt("exhausted handler waiting");
            try {
                wait.acquire();
            } catch (InterruptedException e) {
                e.printStackTrace();
                DebugOut.assertFalse("don't interrupt threads.");
            }
            DebugOut.print_mt("done waiting");
        }

        public void set_synthesis_complete() {
            synthesis_complete.set(true);
            wait.release(local_synthesis.length - 1);
        }

        public void throw_if_synthesis_complete() {
            if (synthesis_complete.get()) {
                throw new ScSynthesisCompleteException();
            }
        }
    }
}
