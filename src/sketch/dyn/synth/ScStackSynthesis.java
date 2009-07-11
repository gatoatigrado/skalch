package sketch.dyn.synth;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScSynthCtrlConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.prefix.ScDefaultPrefix;
import sketch.dyn.prefix.ScPrefixSearchManager;
import sketch.ui.ScUserInterface;
import sketch.util.DebugOut;
import sketch.util.MTReachabilityCheck;

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
public class ScStackSynthesis extends ScSynthesis {
    protected ScLocalStackSynthesis[] local_synthesis;
    protected ScSynthCtrlConf ctrls;
    protected ScSolvingInputConf oracle_inputs;
    protected ScUserInterface ui;
    protected long nsolutions_found = 0;
    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> search_manager;
    public ScExhaustedWaitHandler wait_handler;
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
    }

    @Override
    public boolean synthesize(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        this.ui = ui;
        wait_handler = new ScExhaustedWaitHandler(local_synthesis.length);
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

    public synchronized void add_solution(ScStack stack, int solution_cost) {
        ui.addStackSolution(stack, solution_cost);
        nsolutions_found += 1;
        if (nsolutions_found == nsolutions_to_find) {
            DebugOut.print_mt("synthesis complete");
            wait_handler.set_synthesis_complete();
        }
    }
}
