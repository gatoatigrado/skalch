package sketch.dyn.ga;

import static ec.util.ThreadLocalMT.mt;

import java.util.LinkedList;
import java.util.Vector;

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
    LinkedList<ScGaIndividual> test_queue = new LinkedList<ScGaIndividual>();
    /** the population */
    Vector<ScGaIndividual> done_queue = new Vector<ScGaIndividual>();
    ScPhenotypeMap phenotype;
    public static float prob_mutation_clone = 0.4f;
    public static float prob_reselect = 0.3f;
    public static int max_population_sz = 4;
    static {
        // TODO - make these adjustable
        DebugOut.print("GA Parameter 'ScPopulation.prob_mutation_clone':",
                prob_mutation_clone);
        DebugOut.print("GA Parameter 'ScPopulation.max_population_sz':",
                max_population_sz);
    }

    /** initialize an empty population */
    public ScPopulation(int spine_length) {
        phenotype = new ScPhenotypeMap(spine_length);
        add(new ScGaIndividual(new ScGenotype(), phenotype));
    }

    public void add(ScGaIndividual individual) {
        individual.reset_fitness();
        test_queue.add(individual);
    }

    protected ScPopulation(ScPhenotypeMap phenotype) {
        this.phenotype = phenotype;
    }

    private void clone_and_mutate(ScGaIndividual individual) {
        ScGaIndividual clone = individual.clone();
        clone.genotype.mutate();
        add(clone);
    }

    private void clone_and_crossover(ScGaIndividual first, ScGaIndividual other)
    {
        ScGaIndividual clone = first.clone();
        clone.genotype.crossover(other.genotype);
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
            if (other.pareto_optimal(selected) ^ select_bad) {
                selected = other;
            }
        }
        return selected;
    }

    @Override
    public ScPopulation clone() {
        Vector<ScGaIndividual> next =
                new Vector<ScGaIndividual>(done_queue.size());
        for (ScGaIndividual elt : done_queue) {
            next.add(elt.clone());
        }
        ScPopulation rv = new ScPopulation(phenotype.clone());
        rv.done_queue = next;
        return rv;
    }
}
