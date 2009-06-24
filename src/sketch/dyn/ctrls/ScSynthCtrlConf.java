package sketch.dyn.ctrls;

import java.util.Arrays;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.util.DebugOut;
import sketch.util.Vector4;

/**
 * Wrapper for an array of holes.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthCtrlConf extends ScCtrlConf {
    public ScStack stack;
    public int log_type;
    public int[] values; // -1 for not accessed
    public int[] untilv;
    public int[] set_cnt;
    public String[] value_string;

    public ScSynthCtrlConf(ScConstructInfo[] hole_info, ScStack stack,
            int log_type)
    {
        this.stack = stack;
        this.log_type = log_type;
        values = new int[hole_info.length];
        untilv = new int[hole_info.length];
        set_cnt = new int[hole_info.length];
        for (int a = 0; a < hole_info.length; a++) {
            int uid = hole_info[a].uid();
            if (values[uid] != 0) {
                DebugOut.assertFalse("double initializing uid", uid);
            }
            values[uid] = -1;
            untilv[uid] = hole_info[a].untilv();
        }
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (int value : values) {
            result *= 3333;
            result += value;
        }
        return result;
    }

    public boolean set(int uid, int v) {
        set_cnt[uid] += 1;
        if (v < untilv[uid]) {
            values[uid] = v;
            return true;
        } else {
            return false;
        }
    }

    private String gen_value_string(float f, int v) {
        float amnt_first, amnt_second, amnt_third;
        if (f < 0.5) {
            amnt_first = 1 - 2 * f;
            amnt_second = 2 * f;
            amnt_third = 0;
        } else {
            amnt_first = 0;
            amnt_second = 2 - 2 * f;
            amnt_third = 2 * f - 1;
        }
        Vector4 first = new Vector4(0.f, 0.f, 1.f, 1.f);
        Vector4 second = new Vector4(0.8f, 0.63f, 0.f, 1.f);
        Vector4 third = new Vector4(1.f, 0.f, 0.f, 1.f);
        Vector4 the_color =
                first.scalar_multiply(amnt_first).add(
                        second.scalar_multiply(amnt_second)).add(
                        third.scalar_multiply(amnt_third));
        String color = the_color.hexColor();
        return "<span style=\"color: " + color + ";\">" + Math.max(0, v)
                + "</span>";
    }

    public void reset_accessed(int uid) {
        values[uid] = -1;
    }

    public void copy_values_from(ScSynthCtrlConf ctrls) {
        for (int a = 0; a < ctrls.values.length; a++) {
            values[a] = ctrls.values[a];
            set_cnt[a] = ctrls.set_cnt[a];
        }
    }

    @Override
    public int getValue(int uid) {
        if (values[uid] == -1) {
            values[uid] = 0;
            stack.add_entry(log_type, uid, 0);
        }
        return values[uid];
    }

    public void generate_value_strings() {
        value_string = new String[set_cnt.length];
        int[] maxv_array = set_cnt.clone();
        Arrays.sort(maxv_array);
        int maxv = maxv_array[maxv_array.length - 1];
        for (int a = 0; a < set_cnt.length; a++) {
            float c = (a) / (2.f * (set_cnt.length - 1));
            c += set_cnt[a] / (2.f * maxv);
            value_string[a] = gen_value_string(c, values[a]);
        }
    }

    @Override
    public String getValueString(int uid) {
        return value_string[uid];
    }
}
