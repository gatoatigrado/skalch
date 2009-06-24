package sketch.dyn.synth;

import java.util.NoSuchElementException;
import java.util.Vector;
import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.inputs.ScInputConf;
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
    protected ScCounterexample[] counterexamples;
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

    public void run(ScInputConf[] inputs) {
        // need to clone these as the ScFixedInputGenerators have sketch-local
        // indices
        counterexamples = ScCounterexample.from_inputs(inputs);
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

    /** currently doesn't do much, no MT */
    public final static int NUM_BLIND_FAST = 16;

    public class SynthesisThread extends Thread {
        ScStack stack;
        boolean exhausted = false;
        public float replacement_probability = 1.f;

        /** @returns true if exhausted (need to wait) */
        public boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                // run the program
                // trycatch doesn't seem slow.
                trycatch: try {
                    ScStats.stats.run_test();
                    // DebugOut.print_mt("running test");
                    for (ScCounterexample counterexample : counterexamples) {
                        ScStats.stats.try_counterexample();
                        counterexample.set_for_sketch(sketch);
                        // ssr.reachability_check.check(sketch);
                        if (!sketch.dysketch_main()) {
                            break trycatch;
                        }
                    }
                    ssr.add_solution(stack);
                    ssr.wait_handler.throw_if_synthesis_complete();
                } catch (ScSynthesisAssertFailure e) {
                    if (ssr.print_exceptions) {
                        e.printStackTrace();
                    }
                }
                // advance the stack (whether it succeeded or not)
                try {
                    if (longest_stack == null
                            || longest_stack.stack.size() < stack.stack.size())
                    {
                        longest_stack = stack.clone();
                    }
                    if (rand.get().nextFloat() < replacement_probability) {
                        random_stacks.add(stack.clone());
                        replacement_probability /= 2.f;
                    }
                    stack.next();
                } catch (ScSearchDoneException e) {
                    DebugOut.print_mt("exhausted local search");
                    return true;
                }
            }
            return false; // not exhausted
        }

        public void run_inner() {
            stack = ssr.search_manager.clone_default_search();
            stack.set_for_synthesis(sketch);
            for (int a = 0; !ssr.wait_handler.synthesis_complete.get(); a +=
                    NUM_BLIND_FAST)
            {
                if (ssr.debug_stop_after != -1 && a >= ssr.debug_stop_after) {
                    ssr.wait_handler.wait_exhausted();
                }
                exhausted = blind_fast_routine();
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
                    // TODO - use search manager more
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
