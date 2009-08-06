package sketch.dyn.synth.ga;

import static sketch.util.DebugOut.assertSlow;

import java.util.HashSet;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.ctrls.ScGaCtrlConf;
import sketch.dyn.constructs.inputs.ScGaInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.ga.base.ScGaSolutionId;
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
    public final int spine_length;
    public HashSet<ScGaSolutionId> solutions = new HashSet<ScGaSolutionId>();
    public ScPopulationManager population_mgr;
    protected ScGaIndividual first_solution;

    public ScGaSynthesis(ScDynamicSketchCall<?>[] sketches,
            BackendOptions be_opts)
    {
        super(be_opts);
        spine_length = be_opts.ga_opts.spine_len;
        local_synthesis = new ScLocalGaSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] =
                    new ScLocalGaSynthesis(sketches[a], this, be_opts, a);
        }
    }

    @Override
    public void synthesize_inner(ScUserInterface ui) {
        population_mgr = new ScPopulationManager();
        first_solution = null;
        ui.addGaSynthesis(this);
        for (ScLocalGaSynthesis local_synth : local_synthesis) {
            local_synth.run();
        }
    }

    public synchronized void add_solution(ScGaIndividual current_individual) {
        if (solutions.add(current_individual.generate_solution_id())) {
            ui.addGaSolution(current_individual);
            if (first_solution == null) {
                first_solution = current_individual.clone();
            }
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

    @Override
    public Object get_solution_tuple() {
        if (first_solution == null) {
            return null;
        } else {
            ScGaCtrlConf ctrl_conf = new ScGaCtrlConf();
            ScGaInputConf oracle_conf = new ScGaInputConf();
            first_solution.reset(ctrl_conf, oracle_conf);
            return new scala.Tuple2<ScGaCtrlConf, ScGaInputConf>(ctrl_conf,
                    oracle_conf);
        }
    }
}
