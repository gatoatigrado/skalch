package sketch.dyn.constructs.ctrls;

import java.util.Vector;

import sketch.dyn.synth.stack.ScStack;
import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScHighlightValues;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.ScRichString;

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
    public ScConstructValueString[] value_string;

    public ScSynthCtrlConf(ScStack stack, int log_type) {
        this.stack = stack;
        this.log_type = log_type;
        values = new int[0];
        untilv = new int[0];
        set_cnt = new int[0];
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

    @Override
    public String toString() {
        String[] values_str = new String[values.length];
        for (int a = 0; a < values.length; a++) {
            values_str[a] = String.valueOf(values[a]);
        }
        return "ScSynthCtrlConf[" + (new ScRichString(", ")).join(values_str)
                + "]";
    }

    public void realloc(int min_length) {
        int next_length = Math.max(min_length, values.length * 2);
        int[] next_values = new int[next_length];
        int[] next_untilv = new int[next_length];
        int[] next_set_cnt = new int[next_length];
        System.arraycopy(values, 0, next_values, 0, values.length);
        System.arraycopy(untilv, 0, next_untilv, 0, untilv.length);
        System.arraycopy(set_cnt, 0, next_set_cnt, 0, set_cnt.length);
        for (int a = values.length; a < next_length; a++) {
            next_values[a] = -1;
            next_untilv[a] = -1;
        }
        values = next_values;
        untilv = next_untilv;
        set_cnt = next_set_cnt;
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

    public void reset_accessed(int uid) {
        values[uid] = -1;
    }

    public void copy_values_from(ScSynthCtrlConf prev) {
        values = prev.values.clone();
        untilv = prev.untilv.clone();
        set_cnt = prev.set_cnt.clone();
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
        Vector<ScHighlightValues.Value> value_arr =
                new Vector<ScHighlightValues.Value>();
        for (int a = 0; a < values.length; a++) {
            ScConstructValue value =
                    new ScConstructValue(Math.max(0, values[a]));
            value_arr.add(new ScHighlightValues.Value(value, set_cnt[a], null));
        }
        value_string = new ScConstructValueString[values.length];
        ScHighlightValues.gen_value_strings(value_arr);
        for (int a = 0; a < values.length; a++) {
            value_string[a] = value_arr.get(a).result;
        }
    }

    @Override
    public ScConstructValueString getValueString(int uid)
            throws ScNoValueStringException
    {
        if (uid >= value_string.length) {
            throw new ScNoValueStringException();
        }
        return value_string[uid];
    }

    @Override
    public int getDynamicValue(int uid, int untilv) {
        if (uid >= values.length) {
            realloc(uid + 1);
        }
        this.untilv[uid] = untilv;
        return getValue(uid);
    }

    @Override
    public int[] getValueArray() {
        return values;
    }
}
