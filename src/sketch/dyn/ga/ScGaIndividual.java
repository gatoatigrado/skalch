package sketch.dyn.ga;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.inputs.ScGaInputConf;
import sketch.util.ScCloneable;

/**
 * pretty much a tuple with a clone function. will not become the equivalent of
 * scstack since multiple configurations evolve together (ScPopulation). Since
 * any fitness data is only applicable to a phenotype and genotype, this class
 * contains that information as well.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaIndividual implements ScCloneable<ScGaIndividual> {
    public ScGenotype genotype;
    public ScPhenotypeMap phenotype;
    public int num_asserts_passed;
    public int num_constructs_accessed;
    public int age;
    public int cost;
    boolean done;

    public ScGaIndividual(ScGenotype genotype, ScPhenotypeMap phenotype) {
        this.genotype = genotype;
        this.phenotype = phenotype;
    }

    @Override
    public String toString() {
        return "ScGaIndividual [" + System.identityHashCode(this) + ", age="
                + age + ", num_asserts_passed=" + num_asserts_passed
                + ", num_constructs_accessed=" + num_constructs_accessed
                + ", genotype=" + genotype + ", values=<<<\n"
                + phenotype.formatValuesString(genotype) + ">>>]";
    }

    @Override
    public ScGaIndividual clone() {
        ScGaIndividual result = new ScGaIndividual(genotype.clone(), phenotype);
        result.num_asserts_passed = num_asserts_passed;
        result.num_constructs_accessed = num_constructs_accessed;
        result.age = age;
        return result;
    }

    public void set_for_synthesis_and_reset(ScDynamicSketch sketch,
            ScGaCtrlConf ctrl_conf, ScGaInputConf oracle_conf)
    {
        ctrl_conf.base = this;
        oracle_conf.base = this;
        oracle_conf.reset_accessed();
        genotype.reset_accessed();
        sketch.ctrl_conf = ctrl_conf;
        sketch.oracle_conf = oracle_conf;
    }

    /**
     * pareto-optimality is a false-dominant metric, such that random
     * individuals have a low probability of being replaced by another one
     * ($this$), unless the other one is better in many ways.
     */
    public boolean pareto_optimal(ScGaIndividual selected) {
        if (age > selected.age) {
            return false;
        } else if (num_constructs_accessed < selected.num_constructs_accessed) {
            return false;
        } else if (num_constructs_accessed > selected.num_constructs_accessed) {
            // NOTE - don't compare costs if this one accessed more values,
            // since skAddCost() calls can be anywhere.
            return true;
        } else if (cost >= selected.cost) {
            return false;
        }
        return true;
    }

    /**
     * called when added to the queue of individuals to test
     * @return
     */
    ScGaIndividual reset_fitness() {
        num_asserts_passed = 0;
        num_constructs_accessed = 0;
        age = 0;
        cost = 0;
        done = false;
        return this;
    }

    public ScGaIndividual set_done(int cost) {
        this.cost = cost;
        done = true;
        return this;
    }

    public int synthGetValue(boolean type, int uid, int subuid, int untilv) {
        num_constructs_accessed += 1;
        int idx = phenotype.get_index(type, uid, subuid);
        return genotype.getValue(idx, untilv);
    }

    public int displayGetValue(boolean type, int uid, int subuid, int untilv) {
        int idx = phenotype.get_index(type, uid, subuid);
        return genotype.getValue(idx, untilv);
    }
}
