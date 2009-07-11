package sketch.dyn.synth;

import java.util.NoSuchElementException;
import java.util.Vector;
import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStats;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.AsyncMTEvent;
import sketch.util.DebugOut;
import ec.util.ThreadLocalMT;

/**
 * Container for a synthesis thread. The actual thread is an inner class because
 * the threads will die between synthesis rounds, whereas the sketch object
 * doesn't need to be deleted.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocalStackSynthesis implements ScUiQueueable {
    protected ScDynamicSketch sketch;
    protected ScStackSynthesis ssr;
    protected ScFixedInputConf[] counterexamples;
    public int uid;
    public SynthesisThread thread;
    public ConcurrentLinkedQueue<ScUiModifier> ui_queue;
    public AsyncMTEvent done_events = new AsyncMTEvent();
    public ScStack longest_stack;
    public Vector<ScStack> random_stacks;
    public static ThreadLocalMT rand = new ThreadLocalMT();

    public ScLocalStackSynthesis(ScDynamicSketch sketch, ScStackSynthesis ssr,
            int uid)
    {
        this.sketch = sketch;
        this.ssr = ssr;
        this.uid = uid;
    }

    public void run(ScSolvingInputConf[] inputs) {
        // need to clone these as the ScFixedInputGenerators have sketch-local
        // indices
        counterexamples = ScFixedInputConf.from_inputs(inputs);
        ui_queue = new ConcurrentLinkedQueue<ScUiModifier>();
        random_stacks = new Vector<ScStack>();
        done_events.reset();
        // really basic stuff for now
        if (thread != null && thread.isAlive()) {
            DebugOut.assertFalse("localsynthesis thead alive");
        }
        thread = new SynthesisThread();
        thread.start();
    }

    public void thread_wait() {
        try {
            thread.join();
        } catch (InterruptedException e) {
            DebugOut.assertFalse("interrupted waiting for ScLocalSynthesis");
        }
    }

    public final static int NUM_BLIND_FAST = 8192;

    public class SynthesisThread extends Thread {
        ScStack stack;
        boolean exhausted = false;
        public float replacement_probability = 1.f;
        protected int nruns = 0, ncounterexamples = 0;

        private void update_stats() {
            ScStats.stats.run_test(nruns);
            ScStats.stats.try_counterexample(ncounterexamples);
            nruns = 0;
            ncounterexamples = 0;
        }

        /**
         * NOTE - keep this in sync with ScDebugSketchRun
         * @returns true if exhausted (need to wait)
         */
        protected boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                boolean force_pop = false;
                // run the program
                // trycatch doesn't seem slow.
                trycatch: try {
                    stack.reset_before_run();
                    sketch.solution_cost = 0;
                    nruns++;
                    // DebugOut.print_mt("running test");
                    for (ScFixedInputConf counterexample : counterexamples) {
                        ncounterexamples++;
                        counterexample.set_input_for_sketch(sketch);
                        // ssr.reachability_check.check(sketch);
                        if (!sketch.dysketch_main()) {
                            break trycatch;
                        }
                    }
                    ssr.add_solution(stack, sketch.solution_cost);
                    ssr.wait_handler.throw_if_synthesis_complete();
                } catch (ScSynthesisAssertFailure e) {
                } catch (ScDynamicUntilvException e) {
                    force_pop = true;
                }
                // advance the stack (whether it succeeded or not)
                try {
                    if (longest_stack == null
                            || longest_stack.stack.size() < stack.stack.size())
                    {
                        longest_stack = stack.clone();
                    }
                    if (rand.get().nextFloat() < replacement_probability) {
                        add_random_stack();
                    }
                    stack.next(force_pop);
                } catch (ScSearchDoneException e) {
                    DebugOut.print_mt("exhausted local search");
                    return true;
                }
            }
            return false; // not exhausted
        }

        /** add to the random stacks list and remove half of it */
        private void add_random_stack() {
            random_stacks.add(stack.clone());
            if (random_stacks.size() > ssr.max_num_random) {
                int length1 = random_stacks.size() / 2;
                for (int c = 0; c < length1; c++) {
                    random_stacks.remove(rand.get().nextInt(
                            random_stacks.size()));
                }
                replacement_probability /= 2.f;
            }
        }

        public void run_inner() {
            stack = ssr.search_manager.clone_default_search();
            stack.set_for_synthesis(sketch);
            for (long a = 0; !ssr.wait_handler.synthesis_complete.get(); a +=
                    NUM_BLIND_FAST)
            {
                if (ssr.debug_stop_after != -1 && a >= ssr.debug_stop_after) {
                    ssr.wait_handler.wait_exhausted();
                }
                //
                // NOTE to readers: main call
                exhausted = blind_fast_routine();
                update_stats();
                ssr.wait_handler.throw_if_synthesis_complete();
                if (!ui_queue.isEmpty()) {
                    ui_queue.remove().setInfo(ScLocalStackSynthesis.this, this,
                            stack);
                }
                if (exhausted) {
                    ssr.wait_handler.wait_exhausted();
                    ssr.wait_handler.throw_if_synthesis_complete();
                    //
                    DebugOut.not_implemented("get next active stack");
                    stack = ssr.search_manager.get_active_prefix();
                    stack.set_for_synthesis(sketch);
                }
            }
        }

        /** doc in images/scala/synth_loop/ */
        @Override
        public void run() {
            try {
                run_inner();
            } catch (ScSynthesisCompleteException e) {
            }
            try {
                while (true) {
                    ui_queue.remove().setInfo(ScLocalStackSynthesis.this, this,
                            stack);
                }
            } catch (NoSuchElementException e) {
                update_stats();
                ui_queue = null;
                done_events.set_done();
            }
        }
    }

    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (ui_queue == null) {
            throw new ScUiQueueableInactive();
        }
        ui_queue.add(m);
    }
}
