package sketch.dyn.ga;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.assertSlow;

import java.util.HashSet;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.ga.base.ScGaSolutionId;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;

/**
 * Analogue of ScStackSynthesis
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaSynthesis extends ScSynthesis<ScLocalGaSynthesis> implements
        ScUiQueueable
{
    public int spine_length;
    public HashSet<ScGaSolutionId> solutions = new HashSet<ScGaSolutionId>();
    public ScPopulationManager population_mgr;

    public ScGaSynthesis(ScDynamicSketch[] sketches) {
        local_synthesis = new ScLocalGaSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] = new ScLocalGaSynthesis(sketches[a], this, a);
        }
        spine_length = beopts().ga_opts.spine_len;
    }

    @Override
    public void synthesize_inner(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        population_mgr = new ScPopulationManager();
        ui.addGaSynthesis(this);
        for (ScLocalGaSynthesis local_synth : local_synthesis) {
            local_synth.run(counterexamples);
        }
    }

    public synchronized void add_solution(ScGaIndividual current_individual) {
        if (solutions.add(current_individual.generate_solution_id())) {
            ui.addGaSolution(current_individual);
            increment_num_solutions();
            assertSlow(solutions.contains(current_individual
                    .generate_solution_id()), "solution not added?");
        }
    }

    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        for (ScLocalGaSynthesis ga_local_synth : local_synthesis) {
            if (ga_local_synth.thread != null
                    && ga_local_synth.thread.isAlive())
            {
                ga_local_synth.queueModifier(m);
                return;
            }
        }
    }
}
