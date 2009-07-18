package sketch.dyn.ga.analysis;

import java.util.HashSet;

import sketch.dyn.BackendOptions;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.ga.base.ScGaSolutionId;
import sketch.dyn.stats.ScStatsMT;

/**
 * stats-only analysis of GA
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GaAnalysis {
    ScGaSolutionId[] recent_evaluations =
            new ScGaSolutionId[BackendOptions.ga_opts.analysis_recent];
    public HashSet<ScGaSolutionId> solutions = new HashSet<ScGaSolutionId>();
    protected int num_repeats = 0;
    protected int num_repeats_recent = 0;
    protected int num_mutate = 0;
    protected int num_crossover = 0;

    public void evaluation_done(ScGaIndividual individual) {
        ScGaSolutionId solution_id = individual.generate_solution_id();
        if (!solutions.add(solution_id)) {
            num_repeats += 1;
        }
        for (ScGaSolutionId recent : recent_evaluations) {
            if (recent != null && recent.equals(solution_id)) {
                num_repeats_recent += 1;
                break;
            }
        }
        for (int a = 0; a + 1 < recent_evaluations.length; a++) {
            recent_evaluations[a] = recent_evaluations[a + 1];
        }
        recent_evaluations[recent_evaluations.length - 1] = solution_id;
    }

    public void update_stats() {
        ScStatsMT.stats_singleton.ga_repeated.add(num_repeats);
        ScStatsMT.stats_singleton.ga_nmutate.add(num_mutate);
        ScStatsMT.stats_singleton.ga_ncrossover.add(num_crossover);
        ScStatsMT.stats_singleton.ga_repeated_recent.add(num_repeats_recent);
        num_repeats = 0;
        num_repeats_recent = 0;
        num_mutate = 0;
        num_crossover = 0;
    }

    public void add_clone(ScGaIndividual individual, ScGaIndividual clone) {
        num_mutate += 1;
    }

    public void add_crossover(ScGaIndividual first, ScGaIndividual other,
            ScGaIndividual clone)
    {
        num_crossover += 1;
    }

    public void death(ScGaIndividual individual) {
    }
}
