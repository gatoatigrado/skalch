package sketch.dyn.synth.ga.analysis;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.print;

import java.util.HashSet;

import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.ga.base.ScGaSolutionId;

/**
 * stats-only analysis of GA
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GaAnalysis {
    ScGaSolutionId[] recent_evaluations =
            new ScGaSolutionId[beopts().ga_opts.analysis_recent];
    public HashSet<ScGaSolutionId> solutions = new HashSet<ScGaSolutionId>();
    protected int num_repeats = 0;
    protected int num_repeats_recent = 0;
    protected int num_mutate = 0;
    protected int num_crossover = 0;
    protected int num_select_individual_other_same = 0;
    protected int num_select_individual_other_optimal = 0;
    protected int num_select_individual_selected_optimal = 0;

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

    public GaAnalysis() {
        print("GA Analysis created (this should only be for debugging)");
    }

    public void update_stats() {
        ScStatsMT.stats_singleton.ga_repeated.add(num_repeats);
        ScStatsMT.stats_singleton.ga_nmutate.add(num_mutate);
        ScStatsMT.stats_singleton.ga_ncrossover.add(num_crossover);
        ScStatsMT.stats_singleton.ga_repeated_recent.add(num_repeats_recent);
        ScStatsMT.stats_singleton.ga_selectind_other_same
                .add(num_select_individual_other_same);
        ScStatsMT.stats_singleton.ga_selectind_other_optimal
                .add(num_select_individual_other_optimal);
        ScStatsMT.stats_singleton.ga_selectind_selected_optimal
                .add(num_select_individual_selected_optimal);
        num_repeats = 0;
        num_repeats_recent = 0;
        num_mutate = 0;
        num_crossover = 0;
        num_select_individual_other_same = 0;
        num_select_individual_other_optimal = 0;
        num_select_individual_selected_optimal = 0;
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

    public void select_individual_other_same() {
        num_select_individual_other_same += 1;
    }

    public void select_individual_other_optimal() {
        num_select_individual_other_optimal += 1;
    }

    public void select_individual_selected_optimal() {
        num_select_individual_selected_optimal += 1;
    }
}
