package sketch.dyn.synth;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.result.ScSynthesisResult;
import sketch.util.DebugOut;
import sketch.util.Profiler;

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

    /** currently doesn't do much, no MT */
    public final static int NUM_BLIND_FAST = 16;

    public class SynthesisThread extends Thread {
        ScStack stack;
        boolean exhausted = false;
        Profiler prof;

        /** @returns true if exhausted (need to wait) */
        public boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                // run the program
                // trycatch doesn't seem slow.
                trycatch:
                try {
                    ScStats.stats.run_test();
                    //prof.set_event(Profiler.ProfileEvent.SynthesisStart);
                    // DebugOut.print_mt("running test");
                    for (ScCounterexample counterexample : counterexamples) {
                        ScStats.stats.try_counterexample();
                        counterexample.set_for_sketch(sketch);
                        if (!sketch.dysketch_main()) {
                            break trycatch;
                        }
                    }
                    //prof.set_event(Profiler.ProfileEvent.SynthesisComplete);
                    DebugOut.print_mt("solution string <<<", sketch
                            .solution_str(), ">>>");
                    ssr.add_solution(stack);
                    ssr.wait_handler.throw_if_synthesis_complete();
                } catch (ScSynthesisAssertFailure e) {
                    if (ssr.print_exceptions) {
                        e.printStackTrace();
                    }
                }

                // advance the stack (whether it succeeded or not)
                try {
                    //prof.set_event(Profiler.ProfileEvent.StackNext);
                    stack.next();
                } catch (ScSearchDoneException e) {
                    DebugOut.print_mt("exhausted local search");
                    return true;
                }
            }
            return false; // not exhausted
        }

        public void run_inner() {
            stack = (ScStack) ssr.search_manager.clone_default_search();
            stack.set_for_synthesis(sketch);
            for (int a = 0; !ssr.wait_handler.synthesis_complete.get(); a += NUM_BLIND_FAST) {
                if (ssr.debug_stop_after != -1 && a >= ssr.debug_stop_after) {
                    ssr.wait_handler.wait_exhausted();
                }
                exhausted = blind_fast_routine();
                ssr.wait_handler.throw_if_synthesis_complete();
                if (exhausted) {
                    ssr.wait_handler.wait_exhausted();
                    ssr.wait_handler.throw_if_synthesis_complete();
                    // TODO - use search manager more
                    stack = ssr.search_manager.get_active_prefix();
                    stack.set_for_synthesis(sketch);
                }
            }
        }

        /** doc in images/scala/synth_loop/ */
        public void run() {
            prof = Profiler.profiler.get();
            try {
                run_inner();
            } catch (ScSynthesisCompleteException e) {
            }
        }
    }
}
