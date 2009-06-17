package sketch.dyn.synth;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ScSynthesis;
import sketch.dyn.ScSynthesisCompleteException;
import sketch.dyn.ctrls.ScCtrlConf;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.prefix.ScDefaultPrefix;
import sketch.dyn.prefix.ScPrefixSearchManager;
import sketch.util.DebugOut;

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
    protected ScCtrlConf ctrls;
    protected ScInputConf oracle_inputs;
    protected int nsolutions_to_find;
    protected int nsolutions_found = 0;

    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> search_manager;
    public ExhaustedWaitHandler wait_handler;

    public ScStackSynthesis(ScDynamicSketch[] sketches) {
        // initialize backends
        local_synthesis = new ScLocalStackSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] = new ScLocalStackSynthesis(sketches[a], this);
        }
        ScDefaultPrefix prefix = new ScDefaultPrefix();
        ScStack stack = new ScStack(sketches[0].get_hole_info(), sketches[0]
                .get_oracle_input_list(), prefix);
        search_manager = new ScPrefixSearchManager<ScStack>(stack, prefix);
        nsolutions_to_find = ScSynthesis.cmdopts.int_("num_solutions");
    }

    public boolean synthesize(ScInputConf[] counterexamples) {
        wait_handler = new ExhaustedWaitHandler();
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            local_synth.run(counterexamples);
        }
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            local_synth.thread_wait();
        }
        return wait_handler.synthesis_complete.get();
    }

    public synchronized void add_solution(ScStack stack) {
        DebugOut.print_mt("solution with stack", stack);
        nsolutions_found += 1;
        if (nsolutions_found == nsolutions_to_find) {
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
                synthesis_complete.set(true);
                wait.release(local_synthesis.length - 1);
                n_exhausted.addAndGet(-(local_synthesis.length));
                return;
            }
            DebugOut.print_mt("exhausted handler waiting");
            try {
                wait.acquire();
            } catch (InterruptedException e) {
                e.printStackTrace();
                DebugOut.assert_(false, "don't interrupt threads.");
            }
            DebugOut.print_mt("done waiting");
        }

        public void set_synthesis_complete() {
            synthesis_complete.set(true);
            wait.release(local_synthesis.length - 1);
        }

        public void throw_synthesis_complete() {
            if (synthesis_complete.get()) {
                throw new ScSynthesisCompleteException();
            }
        }
    }
}
