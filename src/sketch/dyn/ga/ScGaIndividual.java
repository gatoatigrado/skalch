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

    public ScGaIndividual(ScGenotype genotype, ScPhenotypeMap phenotype) {
        this.genotype = genotype;
        this.phenotype = phenotype;
    }

    @Override
    public String toString() {
        return "ScGaIndividual [age=" + age + ", genotype=" + genotype
                + ", num_asserts_passed=" + num_asserts_passed
                + ", num_constructs_accessed=" + num_constructs_accessed
                + ", phenotype=" + phenotype + "]";
    }

    @Override
    public ScGaIndividual clone() {
        ScGaIndividual result = new ScGaIndividual(genotype, phenotype);
        result.num_asserts_passed = num_asserts_passed;
        result.num_constructs_accessed = num_constructs_accessed;
        result.age = age;
        return result;
    }

    public void set_for_synthesis(ScDynamicSketch sketch,
            ScGaCtrlConf ctrl_conf, ScGaInputConf oracle_conf)
    {
        ctrl_conf.base = this;
        oracle_conf.base = this;
        oracle_conf.reset_accessed();
        sketch.ctrl_conf = ctrl_conf;
        sketch.oracle_conf = oracle_conf;
    }

    /**
     * NOTE - features could be a linear combination, e.g. f_0 =
     * num_constructs_accessed + num_asserts_passed
     */
    public boolean pareto_optimal(ScGaIndividual selected) {
        return (age < selected.age)
                && (num_constructs_accessed > selected.num_constructs_accessed);
    }

    /** called when added to the queue of individuals to test */
    public void reset_fitness() {
        num_asserts_passed = 0;
        num_constructs_accessed = 0;
        age = 0;
    }

    public int getValue(boolean type, int uid, int subuid, int untilv) {
        num_constructs_accessed += 1;
        int idx = phenotype.get_index(type, uid, subuid);
        return genotype.getValue(idx, untilv);
    }
}
