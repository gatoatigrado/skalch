package sketch.dyn.synth.ga;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.print_mt;
import static sketch.util.ScArrayUtil.deep_clone;

import java.util.LinkedList;
import java.util.Vector;

import sketch.dyn.BackendOptions;
import sketch.dyn.synth.ga.ScGaOptions.ScGaParameter;
import sketch.dyn.synth.ga.analysis.GaAnalysis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.ga.base.ScGenotype;
import sketch.dyn.synth.ga.base.ScPhenotypeMap;
import sketch.util.DebugOut;
import sketch.util.ScCloneable;
import ec.util.MersenneTwisterFast;

/**
 * a population of genotype/phenotypes. there are typically many of these for
 * less communication and less early convergence (greater diversity).
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPopulation implements ScCloneable<ScPopulation> {
    private LinkedList<ScGaIndividual> test_queue =
            new LinkedList<ScGaIndividual>();
    /** the population */
    private Vector<ScGaIndividual> done_queue = new Vector<ScGaIndividual>();
    ScPhenotypeMap phenotype;
    public ScGaParameter prob_clone_mutate;
    public ScGaParameter prob_reselect;
    public int uid;
    public static boolean print_pareto_optimal;
    public static int population_sz;

    /** initialize a population with a single individual */
    public ScPopulation(int spine_length, BackendOptions be_opts) {
        phenotype = new ScPhenotypeMap(spine_length);
        prob_clone_mutate = be_opts.ga_opts.prob_clone_mutate.clone();
        prob_reselect = be_opts.ga_opts.prob_reselect.clone();
        population_sz = be_opts.ga_opts.population_sz;
        print_pareto_optimal = be_opts.ga_opts.print_pareto_optimal;
        for (int a = 0; a < population_sz; a++) {
            add(new ScGaIndividual(this, new ScGenotype(), phenotype));
        }
        uid = mt().nextInt(100000);
    }

    @Override
    public String toString() {
        return String.format("ScPopulation [config= mut %6.2f, resel %6.2f]",
                prob_clone_mutate.value, prob_reselect.value);
    }

    protected ScPopulation(ScPhenotypeMap phenotype) {
        this.phenotype = phenotype;
        uid = mt().nextInt(100000);
    }

    @Override
    public ScPopulation clone() {
        ScPopulation rv = new ScPopulation(phenotype.clone());
        rv.done_queue = deep_clone(done_queue);
        rv.prob_clone_mutate = prob_clone_mutate.clone();
        rv.prob_reselect = prob_reselect.clone();
        return rv;
    }

    public void perturb_parameters() {
        prob_clone_mutate.perturb();
        prob_reselect.perturb();
        print_mt("new parameters", this);
    }

    public void add(ScGaIndividual individual) {
        individual.reset_fitness();
        test_queue.add(individual);
    }

    public boolean test_queue_nonempty() {
        return !test_queue.isEmpty();
    }

    public ScGaIndividual get_individual_for_testing() {
        return test_queue.remove().reset_fitness();
    }

    public void add_done(ScGaIndividual individual) {
        if (!individual.done) {
            DebugOut
                    .assertFalse("call individual.set_done() before add_done()");
        }
        done_queue.add(individual);
    }

    private void clone_and_mutate(ScGaIndividual individual, GaAnalysis analysis)
    {
        ScGaIndividual clone = individual.clone();
        clone.genotype.mutate();
        if (analysis != null) {
            analysis.add_clone(individual, clone);
        }
        add(clone);
    }

    private void clone_and_crossover(ScGaIndividual first,
            ScGaIndividual other, GaAnalysis analysis)
    {
        ScGaIndividual clone = first.clone();
        boolean changed = clone.genotype.crossover(other.genotype);
        if (analysis != null) {
            analysis.add_crossover(first, other, clone);
        }
        if (changed) {
            add(clone);
        } else {
            add_done(clone);
        }
    }

    public void generate_new_phase(GaAnalysis analysis) {
        MersenneTwisterFast mt_local = mt();
        for (int a = 0; a < population_sz; a++) {
            if (done_queue.size() <= 1) {
                clone_and_mutate(done_queue.get(0), analysis);
                return;
            } else {
                ScGaIndividual first =
                        select_individual(mt_local, analysis, false);
                if (mt_local.nextFloat() < prob_clone_mutate.value) {
                    clone_and_mutate(first, analysis);
                } else {
                    ScGaIndividual other =
                            select_individual(mt_local, analysis, false);
                    if (other == first) {
                        clone_and_mutate(first, analysis);
                    } else {
                        clone_and_crossover(first, other, analysis);
                    }
                }
            }
        }
    }

    public void death_phase(GaAnalysis analysis) {
        int to_kill = done_queue.size() - population_sz;
        MersenneTwisterFast mt_local = mt();
        for (int a = 0; a < to_kill; a++) {
            ScGaIndividual individual =
                    select_individual(mt_local, analysis, true);
            if (analysis != null) {
                analysis.death(individual);
            }
            if (!done_queue.remove(individual)) {
                DebugOut.assertFalse("couldn't remove element");
            }
        }
        // increase the age of everything still alive
        for (ScGaIndividual individual : done_queue) {
            individual.age += 1;
        }
    }

    /**
     * @param exclude
     *            exclude selecting this individual
     */
    private ScGaIndividual select_individual(MersenneTwisterFast mt_local,
            GaAnalysis analysis, boolean select_bad)
    {
        ScGaIndividual selected =
                done_queue.get(mt_local.nextInt(done_queue.size()));
        while (mt_local.nextFloat() < prob_reselect.value) {
            ScGaIndividual other =
                    done_queue.get(mt_local.nextInt(done_queue.size()));
            if (other == selected) {
                if (analysis != null && !select_bad) {
                    analysis.select_individual_other_same();
                }
                break;
            } else if (other.pareto_optimal(selected) ^ select_bad) {
                if (analysis != null && !select_bad) {
                    analysis.select_individual_other_optimal(other, selected);
                }
                selected = other;
            } else if (analysis != null && !select_bad) {
                analysis.select_individual_selected_optimal(selected, other);
            }
        }
        return selected;
    }
}
