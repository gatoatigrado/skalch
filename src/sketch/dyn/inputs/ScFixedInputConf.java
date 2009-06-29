package sketch.dyn.inputs;

import java.util.Vector;

import sketch.dyn.ScDynamicSketch;
import sketch.ui.sourcecode.ScHighlightValues;
import sketch.util.DebugOut;
import sketch.util.RichString;

/**
 * static inputs that may have slightly faster access
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScFixedInputConf extends ScInputConf {
    protected int[][] values;
    protected int[][] set_cnt;
    protected int[] untilv;
    protected int[] next;
    public String[] value_string;

    public ScFixedInputConf(int[][] values, int[][] set_cnt, int[] untilv,
            int[] next)
    {
        this.values = values;
        this.set_cnt = set_cnt;
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
        sketch.input_conf = this;
        reset_index();
    }

    /** NOTE - a bit of messy (uid, subuid) -> index mapping */
    public void generate_value_strings() {
        Vector<ScHighlightValues.Value> value_arr =
                new Vector<ScHighlightValues.Value>();
        for (int uid_idx = 0; uid_idx < values.length; uid_idx++) {
            for (int subuid_idx = 0; subuid_idx < values[uid_idx].length; subuid_idx++)
            {
                String value = String.valueOf(values[uid_idx][subuid_idx]);
                int color_v = set_cnt[uid_idx][subuid_idx];
                int[] id_arr = { uid_idx, subuid_idx };
                value_arr.add(new ScHighlightValues.Value(value, color_v,
                        id_arr));
            }
        }
        ScHighlightValues.gen_value_strings(value_arr);
        // convert to a 2-d vector
        Vector<Vector<String>> results = new Vector<Vector<String>>();
        for (ScHighlightValues.Value value : value_arr) {
            int[] id_arr = (int[]) value.tag;
            if (id_arr[0] >= results.size()) {
                results.setSize(id_arr[0] + 1);
                results.set(id_arr[0], new Vector<String>());
            }
            Vector<String> uid_v = results.get(id_arr[0]);
            if (id_arr[1] >= uid_v.size()) {
                uid_v.setSize(id_arr[1] + 1);
            }
            uid_v.set(id_arr[1], value.result);
        }
        // finally generate strings
        value_string = new String[values.length];
        for (int a = 0; a < value_string.length; a++) {
            if (results.get(a) != null) {
                value_string[a] =
                        (new RichString(", ")).join(results.get(a).toArray(
                                new String[0]));
            } else {
                value_string[a] = "/* not encountered */";
            }
        }
    }

    @Override
    public String getValueString(int uid) {
        return value_string[uid];
    }
}
