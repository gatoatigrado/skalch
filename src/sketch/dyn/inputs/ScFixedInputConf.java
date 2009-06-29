package sketch.dyn.inputs;

import sketch.dyn.ScDynamicSketch;
import sketch.util.DebugOut;

/**
 * static inputs that may have slightly faster access
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScFixedInputConf extends ScInputConf {
    protected int[][] values;
    protected int[] untilv;
    protected int[] next;
    public String[] value_string;

    public ScFixedInputConf(int[][] values, int[] untilv, int[] next) {
        this.values = values;
        this.untilv = untilv;
        this.next = next;
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (int[] v_arr : values) {
            result *= 13;
            for (int v : v_arr) {
                result *= 344;
                result += v;
            }
        }
        return result;
    }

    @Override
    public int dynamicNextValue(int uid, int untilv_) {
        if (untilv[uid] != untilv_) {
            DebugOut.assertFalse("set untilv for fixed input generators");
        }
        return nextValue(uid);
    }

    @Override
    public int nextValue(int uid) {
        final int uid_next = next[uid];
        if (uid_next >= values[uid].length) {
            DebugOut.assertFalse("fixed input generator exceeding length", uid,
                    uid_next, values[uid].length);
            return 0;
        } else {
            int rv = values[uid][uid_next];
            next[uid] += 1;
            return rv;
        }
    }

    public static ScFixedInputConf[] from_inputs(ScSolvingInputConf[] inputs) {
        ScFixedInputConf[] result = new ScFixedInputConf[inputs.length];
        for (int a = 0; a < inputs.length; a++) {
            result[a] = inputs[a].fixed_inputs();
        }
        return result;
    }

    public void reset_index() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }

    public void set_for_sketch(ScDynamicSketch sketch) {
        sketch.input_backend = this;
        reset_index();
    }

    @Override
    public String getValueString(int uid) {
        return value_string[uid];
    }
}
