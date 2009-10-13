package sketch.dyn.constructs.ctrls;

import java.util.Vector;

import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.ga.base.ScPhenotypeEntry;
import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;

/**
 * proxy class which will call into a ScGaIndividual
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaCtrlConf extends ScCtrlConf {
    public ScGaIndividual base;
    public int[] default_untilv = new int[0];

    @Override
    public int getDynamicValue(int uid, int untilv) {
        return base.synthGetValue(true, uid, 0, untilv);
    }

    @Override
    public int getValue(int uid) {
        return base.synthGetValue(true, uid, 0, default_untilv[uid]);
    }

    @Override
    public ScConstructValueString getValueString(int uid)
            throws ScNoValueStringException
    {
        ScConstructValue value =
                new ScConstructValue(base
                        .displayGetValue(true, uid, 0, 1 << 20));
        return new ScConstructValueString("", value, "");
    }

    @Override
    public int[] getValueArray() {
        Vector<ScPhenotypeEntry> activeCtrls =
                base.phenotype.getActiveEntriesOfType(true);
        int[] result = new int[activeCtrls.size()];
        for (int a = 0; a < activeCtrls.size(); a++) {
            result[a] =
                    base.genotype.getValue(activeCtrls.get(a).index, 1 << 50);
        }
        return result;
    }
}
