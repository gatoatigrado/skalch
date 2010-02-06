package sketch.dyn.synth.ga.analysis;

import static sketch.util.DebugOut.print;

import java.util.HashSet;

import sketch.dyn.BackendOptions;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.ga.base.ScGaSolutionId;
import sketch.util.DebugOut;

/**
 * stats-only analysis of GA
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GaAnalysis {
    protected final ScGaSolutionId[] recent_evaluations;
    public final HashSet<ScGaSolutionId> solutions =
            new HashSet<ScGaSolutionId>();
    protected int num_repeats = 0;
    protected int num_repeats_recent = 0;
    protected int num_mutate = 0;
    protected int num_crossover = 0;
    protected int num_select_individual_other_same = 0;
    protected int num_select_individual_other_optimal = 0;
    protected int num_select_individual_selected_optimal = 0;
    protected final boolean print_pareto_optimal;

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

    public GaAnalysis(BackendOptions be_opts) {
        print("GA Analysis created (this should only be for debugging)");
        recent_evaluations =
                new ScGaSolutionId[be_opts.ga_opts.analysis_recent];
        print_pareto_optimal = be_opts.ga_opts.print_pareto_opt;
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

    public void select_individual_other_optimal(ScGaIndividual optimal,
            ScGaIndividual suboptimal)
    {
        num_select_individual_other_optimal += 1;
        print_optimal(optimal, suboptimal);
    }

    public void select_individual_selected_optimal(ScGaIndividual optimal,
            ScGaIndividual suboptimal)
    {
        num_select_individual_selected_optimal += 1;
        print_optimal(optimal, suboptimal);
    }

    public void print_optimal(ScGaIndividual optimal, ScGaIndividual suboptimal)
    {
        if (print_pareto_optimal) {
            DebugOut.print_mt("individual " + optimal + "\noptimal to\n"
                    + suboptimal + "\n");
        }
    }
}
