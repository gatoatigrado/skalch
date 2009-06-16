package sketch.dyn.synth;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.synth.result.ScSynthesisResult;
import sketch.util.DebugOut;

/**
 * Container for a synthesis thread. The actual thread is an inner class because
 * the threads will die between synthesis rounds, whereas the sketch object
 * doesn't need to be deleted.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocalStackSynthesis {
    protected ScDynamicSketch sketch;
    protected ScStackSynthesis ssr;
    protected ScCounterexample[] counterexamples;
    public SynthesisThread thread;
    public ScSynthesisResult synthesis_result;

    public ScLocalStackSynthesis(ScDynamicSketch sketch, ScStackSynthesis ssr) {
        this.sketch = sketch;
        this.ssr = ssr;
    }

    public void run(ScInputConf[] inputs) {
        // need to clone these as the ScFixedInputGenerators have sketch-local
        // indices
        counterexamples = new ScCounterexample[inputs.length];
        for (int a = 0; a < inputs.length; a++) {
            counterexamples[a] = new ScCounterexample(inputs[a].fixed_inputs());
        }
        synthesis_result = null;

        // really basic stuff for now
        DebugOut.assert_(thread == null || !thread.isAlive(),
                "localsynthesis thead alive");
        thread = new SynthesisThread();
        thread.start();
    }

    public void thread_wait() {
        try {
            thread.join();
        } catch (InterruptedException e) {
            DebugOut.assert_(false, "interrupted waiting for ScLocalSynthesis");
        }
    }

    // TODO - experiment, set to twice min
    public final static int NUM_BLIND_FAST = 1;

    public class SynthesisThread extends Thread {
        ScStack stack;
        boolean exhausted = false;

        /** @returns true if exhausted (need to wait) */
        public boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                // run the program
                try {
                    DebugOut.print_mt("running test");
                    for (ScCounterexample counterexample : counterexamples) {
                        counterexample.set_for_sketch(sketch);
                        sketch.dysketch_main();
                    }
                    ssr.add_solution(stack);
                    if (ssr.wait_handler.synthesis_complete.get()) {
                        // everyone's exiting, order doesn't matter
                        return false;
                    }
                } catch (Exception e) {
                } catch (java.lang.AssertionError e) {
                }

                // advance the stack (whether it succeeded or not)
                try {
                    stack.next();
                } catch (ScSearchDoneException e) {
                    DebugOut.print_mt("exhausted local search");
                    return true;
                }
            }
            return false; // not exhausted
        }

        /** doc in images/scala/synth_loop/ */
        public void run() {
            stack = (ScStack) ssr.search_manager.clone_default_search();
            stack.set_for_synthesis(sketch);
            while (!ssr.wait_handler.synthesis_complete.get()) {
                exhausted = blind_fast_routine();
                if (ssr.wait_handler.synthesis_complete.get()) {
                    return;
                } else if (exhausted) {
                    ssr.wait_handler.wait_exhausted();
                    stack = ssr.search_manager.get_active_prefix();
                    stack.set_for_synthesis(sketch);
                }
            }
        }
    }
}
