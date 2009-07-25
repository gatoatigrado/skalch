package sketch.dyn.stack;

import java.util.concurrent.atomic.AtomicBoolean;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScSynthCtrlConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.prefix.ScDefaultPrefix;
import sketch.dyn.prefix.ScPrefixSearchManager;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;

/**
 * cloned Lexin's implementation, then modified for
 * <ul>
 * <li>support for oracle values (!!)</li>
 * <li>backtracking / exploration in multiple regions with multithreading</li>
 * </ul>
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStackSynthesis extends ScSynthesis<ScLocalStackSynthesis> {
    protected ScSynthCtrlConf ctrls;
    protected ScSolvingInputConf oracle_inputs;
    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> search_manager;
    protected AtomicBoolean got_first_run;

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
                        .get_oracle_info(), prefix);
        // shared classes to synchronize / manage search
        search_manager = new ScPrefixSearchManager<ScStack>(stack, prefix);
        got_first_run = new AtomicBoolean(false);
    }

    @Override
    public void synthesize_inner(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        for (ScLocalStackSynthesis local_synth : local_synthesis) {
            ui.addStackSynthesis(local_synth);
            local_synth.run(counterexamples);
        }
    }

    public void add_solution(ScStack stack, int solution_cost) {
        if (stack.first_run && got_first_run.getAndSet(true)) {
            return;
        }
        ui.addStackSolution(stack, solution_cost);
        increment_num_solutions();
    }
}
