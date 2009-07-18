package sketch.dyn.inputs;

import static sketch.util.ScArrayUtil.extend_arr;

import java.util.Vector;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.ScCloneable;

/**
 * proxy class which will call into a ScGaIndividual
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaInputConf extends ScInputConf implements
        ScCloneable<ScGaInputConf>
{
    public ScGaIndividual base;
    public int[] next;
    public int[] default_untilv;

    public ScGaInputConf(ScConstructInfo[] oracle_info) {
        next = new int[oracle_info.length];
        default_untilv = new int[oracle_info.length];
        for (int a = 0; a < oracle_info.length; a++) {
            default_untilv[oracle_info[a].uid()] = oracle_info[a].untilv();
        }
    }

    protected ScGaInputConf(int[] next, int[] default_untilv) {
        this.next = next;
        this.default_untilv = default_untilv;
    }

    @Override
    public int dynamicNextValue(int uid, int untilv) {
        if (uid >= next.length) {
            realloc(2 * uid + 1);
        }
        int rv = base.synthGetValue(false, uid, next[uid], untilv);
        next[uid] += 1;
        return rv;
    }

    @Override
    public Vector<ScConstructValueString> getValueString(int uid)
            throws ScNoValueStringException
    {
        Vector<ScConstructValueString> result =
                new Vector<ScConstructValueString>();
        if (uid >= next.length) {
            return result;
        }
        for (int subuid = 0; subuid < next[uid]; subuid++) {
            ScConstructValue value =
                    new ScConstructValue(base.displayGetValue(false, uid, 0,
                            1 << 20));
            result.add(new ScConstructValueString("", value, ""));
        }
        return result;
    }

    @Override
    public int nextValue(int uid) {
        int rv = base.synthGetValue(false, uid, next[uid], default_untilv[uid]);
        next[uid] += 1;
        return rv;
    }

    protected void realloc(int length) {
        next = extend_arr(next, length);
        default_untilv = extend_arr(next, length, 10);
    }

    @Override
    public ScGaInputConf clone() {
        return new ScGaInputConf(next.clone(), default_untilv.clone());
    }

    public void reset_accessed() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }
}
