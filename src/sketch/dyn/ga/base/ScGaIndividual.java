package sketch.dyn.ga.base;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.ga.base.ScGaSolutionId.ScGaSolutionIdEntry;
import sketch.dyn.inputs.ScGaInputConf;
import sketch.util.ScCloneable;
import sketch.util.ScHtmlUtil;

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
    public int age;
    public int cost;
    public int solution_id_hash = -1;
    public boolean done;

    public ScGaIndividual(ScGenotype genotype, ScPhenotypeMap phenotype) {
        this.genotype = genotype;
        this.phenotype = phenotype;
    }

    @Override
    public String toString() {
        return "ScGaIndividual [" + System.identityHashCode(this) + ", age="
                + age + ", num_asserts_passed=" + num_asserts_passed
                + ", genotype=" + genotype + ", values=<<<\n"
                + phenotype.formatValuesString(genotype) + ">>>]";
    }

    public String valuesString() {
        return phenotype.formatValuesString(genotype);
    }

    /**
     * NOTE - doesn't change the phenotype, but that should only be adding
     * entries, so it shouldn't change the semantics of the genotype.
     */
    @Override
    public ScGaIndividual clone() {
        ScGaIndividual result = new ScGaIndividual(genotype.clone(), phenotype);
        result.num_asserts_passed = num_asserts_passed;
        result.age = age;
        result.solution_id_hash = solution_id_hash;
        return result;
    }

    /** resets accessed bits in the genotype and resets the oracle configuration */
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
        } else if (num_asserts_passed < selected.num_asserts_passed) {
            return false;
        } else if (num_asserts_passed > selected.num_asserts_passed) {
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
    public ScGaIndividual reset_fitness() {
        num_asserts_passed = 0;
        age = 0;
        cost = 0;
        done = false;
        return this;
    }

    public ScGaIndividual set_done(int cost, int num_asserts_passed) {
        this.cost = cost;
        this.num_asserts_passed = num_asserts_passed;
        done = true;
        return this;
    }

    public int synthGetValue(boolean type, int uid, int subuid, int untilv) {
        int idx = phenotype.get_index(type, uid, subuid);
        return genotype.getValue(idx, untilv);
    }

    public int displayGetValue(boolean type, int uid, int subuid, int untilv) {
        int idx = phenotype.get_index(type, uid, subuid);
        return genotype.getValue(idx, untilv);
    }

    public ScGaSolutionId generate_solution_id() {
        ScGaSolutionId result = new ScGaSolutionId();
        for (int a = 0; a < phenotype.entries.size()
                && a < genotype.data.length; a++)
        {
            ScPhenotypeEntry entry = phenotype.entries.get(a);
            if (entry != null && genotype.active_data[a]) {
                result.entries.add(new ScGaSolutionIdEntry(entry.type,
                        entry.uid, entry.subuid, genotype.data[a]));
            }
        }
        result.create_array();
        solution_id_hash = result.hashCode();
        return result;
    }

    public String htmlDebugString() {
        return ScHtmlUtil.html_nonpre_code(toString());
    }
}
