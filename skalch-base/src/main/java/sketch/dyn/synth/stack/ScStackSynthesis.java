package sketch.dyn.synth.stack;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.ctrls.ScSynthCtrlConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.stack.prefix.ScDefaultPrefix;
import sketch.dyn.synth.stack.prefix.ScPrefixSearchManager;
import sketch.ui.ScUserInterface;
import sketch.ui.queues.Queue;

/**
 * cloned Lexin's implementation, then modified for
 * <ul>
 * <li>support for oracle values (!!)</li>
 * <li>backtracking / exploration in multiple regions with multithreading</li>
 * </ul>
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStackSynthesis extends ScSynthesis<ScLocalStackSynthesis> {
    protected ScSynthCtrlConf ctrls;
    protected ScSolvingInputConf oracle_inputs;
    protected Queue queue;

    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> search_manager;
    protected AtomicBoolean got_first_run;
    protected AtomicReference<ScStack> first_solution;

    public ScStackSynthesis(ScDynamicSketchCall<?>[] sketches,
            BackendOptions be_opts) {
        super(be_opts);
        // initialize backends
        local_synthesis = new ScLocalStackSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] = new ScLocalStackSynthesis(sketches[a], this,
                    be_opts, a);
        }
    }

    @Override
    public void synthesize_inner(ScUserInterface ui) {
        ScDefaultPrefix prefix = new ScDefaultPrefix();
        ScStack stack = new ScStack(prefix, be_opts.synth_opts.max_stack_depth);
        // shared classes to synchronize / manage search
        search_manager = new ScPrefixSearchManager<ScStack>(stack, prefix);
        got_first_run = new AtomicBoolean(false);
        first_solution = new AtomicReference<ScStack>(null);
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            ui.addStackSynthesis(local_synth);
            local_synth.run();
        }
    }

    public boolean add_solution(ScStack stack) {
        if (stack.first_run && got_first_run.getAndSet(true)) {
            return false;
        }
        if (first_solution.get() == null) {
            first_solution.compareAndSet(null, stack.clone());
        }
        ui.addStackSolution(stack);
        increment_num_solutions();
        return true;
    }

    @Override
    public Object get_solution_tuple() {
        if (first_solution.get() == null) {
            return null;
        } else {
            ScStack stack = first_solution.get();
            stack.reset_before_run(); // let the solution just dequeue entries
            return new scala.Tuple2<ScSynthCtrlConf, ScSolvingInputConf>(
                    stack.ctrl_conf, stack.oracle_conf);
        }
    }
}
