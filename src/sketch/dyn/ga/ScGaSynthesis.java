package sketch.dyn.ga;

import sketch.dyn.BackendOptions;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;

/**
 * Analogue of ScStackSynthesis
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaSynthesis extends ScSynthesis<ScLocalGaSynthesis> {
    public int spine_length;

    public ScGaSynthesis(ScDynamicSketch[] sketches) {
        local_synthesis = new ScLocalGaSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] = new ScLocalGaSynthesis(sketches[a], this, a);
        }
        spine_length = (int) BackendOptions.ga_opts.long_("spine_len");
    }

    @Override
    public void synthesize_inner(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        ui.addGaSynthesis(this);
        for (ScLocalGaSynthesis local_synth : local_synthesis) {
            local_synth.run(counterexamples);
        }
    }

    public void add_solution(ScGaIndividual current_individual) {
        ui.addGaSolution(current_individual);
        increment_num_solutions();
    }
}
