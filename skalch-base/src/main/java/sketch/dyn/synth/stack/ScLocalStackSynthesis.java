package sketch.dyn.synth.stack;

import java.util.Vector;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScLocalSynthesis;
import sketch.dyn.synth.ScSearchDoneException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
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
public class ScLocalStackSynthesis extends ScLocalSynthesis {
    protected ScStackSynthesis ssr;
    public int longest_stack_size;
    public ScStack longest_stack;
    public Vector<ScStack> random_stacks;

    public ScLocalStackSynthesis(ScDynamicSketchCall<?> sketch,
            ScStackSynthesis ssr, int uid)
    {
        super(sketch, uid);
        this.ssr = ssr;
    }

    @Override
    public ScStackSynthesisThread create_synth_thread() {
        random_stacks = new Vector<ScStack>();
        return new ScStackSynthesisThread();
    }

    public class ScStackSynthesisThread extends AbstractSynthesisThread {
        ScStack stack;
        boolean exhausted = false;
        public float replacement_probability = 1.f;

        /**
         * NOTE - keep this in sync with ScDebugSketchRun
         * @returns true if exhausted (need to wait)
         */
        protected boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                boolean force_pop = false;
                trycatch: try {
                    stack.reset_before_run();
                    sketch.initialize_before_all_tests(stack.ctrl_conf,
                            stack.oracle_conf);
                    nruns += 1;
                    for (int c = 0; c < sketch.get_num_counterexamples(); c++) {
                        ncounterexamples += 1;
                        if (!sketch.run_test(c)) {
                            break trycatch;
                        }
                    }
                    nsolutions += 1;
                    stack.setCost(sketch.get_solution_cost());
                    ssr.add_solution(stack);
                    ssr.wait_handler.throw_if_synthesis_complete();
                } catch (ScSynthesisAssertFailure e) {
                } catch (ScDynamicUntilvException e) {
                    force_pop = true;
                }
                // advance the stack (whether it succeeded or not)
                try {
                    if (longest_stack_size < stack.stack.size()) {
                        longest_stack = stack.clone();
                        longest_stack_size = stack.stack.size();
                    }
                    if (mt_local.nextFloat() < replacement_probability) {
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
                    random_stacks
                            .remove(mt_local.nextInt(random_stacks.size()));
                }
                replacement_probability /= 2.f;
            }
        }

        @Override
        public void run_inner() {
            longest_stack_size = -1;
            longest_stack = null;
            stack = ssr.search_manager.clone_default_search();
            for (long a = 0; !ssr.wait_handler.synthesis_complete.get(); a +=
                    NUM_BLIND_FAST)
            {
                if (a >= ssr.debug_stop_after) {
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
                }
            }
        }

        @Override
        public void process_ui_queue(ScUiModifier ui_modifier) {
            ui_modifier.setInfo(ScLocalStackSynthesis.this, this, stack);
        }
    }

    @Override
    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (ui_queue == null) {
            throw new ScUiQueueableInactive();
        }
        ui_queue.add(m);
    }
}
