package sketch.dyn.inputs;

import static sketch.util.ScArrayUtil.extend_arr;

import java.util.Vector;

import sketch.dyn.ga.ScGaIndividual;
import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.ScCloneable;

public class ScGaInputConf extends ScInputConf implements
        ScCloneable<ScGaInputConf>
{
    public ScGaIndividual base;
    public int[] next;
    public int[] default_untilv;

    public ScGaInputConf(int[] default_untilv) {
        next = new int[default_untilv.length];
        this.default_untilv = default_untilv;
    }

    @Override
    public int dynamicNextValue(int uid, int untilv) {
        if (uid >= next.length) {
            realloc(2 * uid + 1);
        }
        return base.getValue(false, uid, next[uid], untilv);
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
                    new ScConstructValue(base.getValue(false, uid, 0, 1 << 20));
            result.add(new ScConstructValueString("", value, ""));
        }
        return result;
    }

    @Override
    public int nextValue(int uid) {
        return base.getValue(false, uid, next[uid], default_untilv[uid]);
    }

    protected void realloc(int length) {
        next = extend_arr(next, length);
        default_untilv = extend_arr(next, length, 10);
    }

    @Override
    public ScGaInputConf clone() {
        ScGaInputConf result = new ScGaInputConf(default_untilv);
        result.next = next.clone();
        return result;
    }

    public void reset_accessed() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }
}
