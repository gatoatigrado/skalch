package sketch.dyn.ga;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.ScArrayUtil.deep_clone;

import java.util.LinkedList;
import java.util.Vector;

import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.ga.base.ScGenotype;
import sketch.dyn.ga.base.ScPhenotypeMap;
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
    public static float prob_mutation_clone = 0.2f;
    public static float prob_crossover_mutate_different = 0.8f;
    public static float prob_reselect = 0.9f;
    public static int max_population_sz = 8;
    static {
        // TODO - make these adjustable
        DebugOut.print("GA Parameter 'ScPopulation.prob_mutation_clone':",
                prob_mutation_clone);
    }

    /** initialize a population with a single individual */
    public ScPopulation(int spine_length) {
        phenotype = new ScPhenotypeMap(spine_length);
        add(new ScGaIndividual(new ScGenotype(), phenotype));
    }

    protected ScPopulation(ScPhenotypeMap phenotype) {
        this.phenotype = phenotype;
    }

    @Override
    public ScPopulation clone() {
        ScPopulation rv = new ScPopulation(phenotype.clone());
        rv.done_queue = deep_clone(done_queue);
        return rv;
    }

    public void perturb_parameters() {
        // ...
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

    private void clone_and_mutate(ScGaIndividual individual) {
        ScGaIndividual clone = individual.clone();
        clone.genotype.mutate();
        add(clone);
    }

    private void clone_and_crossover(ScGaIndividual first, ScGaIndividual other)
    {
        ScGaIndividual clone = first.clone();
        clone.genotype.crossover(other.genotype,
                prob_crossover_mutate_different);
        add(clone);
    }

    public void generate_new_phase() {
        MersenneTwisterFast mt_local = mt();
        for (int a = 0; a < max_population_sz; a++) {
            if (done_queue.size() <= 1) {
                clone_and_mutate(done_queue.get(0));
                return;
            } else {
                ScGaIndividual first = select_individual(mt_local, false);
                if (mt_local.nextFloat() < prob_mutation_clone) {
                    clone_and_mutate(first);
                } else {
                    ScGaIndividual other = select_individual(mt_local, false);
                    if (other == first) {
                        clone_and_mutate(first);
                    } else {
                        clone_and_crossover(first, other);
                    }
                }
            }
        }
    }

    public void death_phase() {
        int to_kill = done_queue.size() - max_population_sz;
        MersenneTwisterFast mt_local = mt();
        for (int a = 0; a < to_kill; a++) {
            if (!done_queue.remove(select_individual(mt_local, true))) {
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
            boolean select_bad)
    {
        ScGaIndividual selected =
                done_queue.get(mt_local.nextInt(done_queue.size()));
        while (mt_local.nextFloat() < prob_reselect) {
            ScGaIndividual other =
                    done_queue.get(mt_local.nextInt(done_queue.size()));
            if (other == selected) {
                break;
            } else if (other.pareto_optimal(selected) ^ select_bad) {
                selected = other;
            }
        }
        return selected;
    }
}
