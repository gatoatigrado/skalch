package sketch.dyn.ctrls;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.ga.ScGaIndividual;
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
    public int[] default_untilv;

    public ScGaCtrlConf(ScConstructInfo[] info) {
        default_untilv = new int[info.length];
        for (int a = 0; a < info.length; a++) {
            default_untilv[info[a].uid()] = info[a].untilv();
        }
    }

    @Override
    public int getDynamicValue(int uid, int untilv) {
        return base.getValue(true, uid, 0, untilv);
    }

    @Override
    public int getValue(int uid) {
        return base.getValue(true, uid, 0, default_untilv[uid]);
    }

    @Override
    public ScConstructValueString getValueString(int uid)
            throws ScNoValueStringException
    {
        ScConstructValue value =
                new ScConstructValue(base.getValue(true, uid, 0, 1 << 20));
        return new ScConstructValueString("", value, "");
    }
}
