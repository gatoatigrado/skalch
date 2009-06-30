package sketch.dyn.inputs;

import java.util.Vector;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.util.DebugOut;

/**
 * Common functions on an array of solving inputs. Most importantly, these can
 * be converted to fixed inputs.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSolvingInputConf extends ScInputConf implements Cloneable {
    protected ScStack stack;
    protected int log_type;
    protected Vector<Integer>[] values;
    protected Vector<Integer>[] set_cnt;
    protected int[] untilv;
    protected int[] next;
    protected int num_uids;

    @SuppressWarnings("unchecked")
    public ScSolvingInputConf(ScConstructInfo[] input_info, ScStack stack,
            int log_type)
    {
        this.stack = stack;
        this.log_type = log_type;
        num_uids = input_info.length;
        values = new Vector[input_info.length]; // can't create generic array
        set_cnt = new Vector[input_info.length];
        untilv = new int[input_info.length];
        next = new int[input_info.length];
        for (int a = 0; a < input_info.length; a++) {
            int uid = input_info[a].uid();
            if (uid >= input_info.length) {
                DebugOut.assertFalse("uid greater than hole info length", uid,
                        input_info.length);
            } else if (values[uid] != null) {
                DebugOut.assertFalse("double initializing uid", uid);
            }
            values[uid] = new Vector<Integer>(10);
            set_cnt[uid] = new Vector<Integer>(10);
            untilv[uid] = input_info[a].untilv();
        }
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (Vector<Integer> input : values) {
            result *= 9221;
            result += input.hashCode();
        }
        return result;
    }

    /** for clone method only */
    protected ScSolvingInputConf() {
    }

    public void add_input(int uid, int v) {
        values[uid].add(v);
    }

    /** copies synthesized inputs to fixed inputs. result is threadsafe / unique */
    public ScFixedInputConf fixed_inputs() {
        int[][] fixed_values = new int[num_uids][];
        int[][] set_cnt_values = new int[num_uids][];
        for (int a = 0; a < num_uids; a++) {
            fixed_values[a] = new int[values[a].size()];
            set_cnt_values[a] = new int[values[a].size()];
            for (int c = 0; c < fixed_values[a].length; c++) {
                fixed_values[a][c] = values[a].get(c);
                if (set_cnt[a].size() <= c || set_cnt[a].get(c) == null) {
                    set_cnt_values[a][c] = 0;
                } else {
                    set_cnt_values[a][c] = set_cnt[a].get(c);
                }
            }
        }
        return new ScFixedInputConf(fixed_values, set_cnt_values, untilv
                .clone(), next.clone());
    }

    public boolean set(int uid, int subuid, int v) {
        if (v < untilv[uid]) {
            if (set_cnt[uid].size() <= subuid) {
                set_cnt[uid].setSize(subuid + 1);
            }
            final Integer prev = set_cnt[uid].get(subuid);
            if (prev == null) {
                set_cnt[uid].set(subuid, 1);
            } else {
                set_cnt[uid].set(subuid, prev + 1);
            }
            values[uid].set(subuid, v);
            return true;
        } else {
            return false;
        }
    }

    public int get(int uid, int subuid) {
        return values[uid].get(subuid);
    }

    public void reset_index() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }

    public void reset_accessed(int uid, int subuid) {
        values[uid].remove(subuid);
    }

    @SuppressWarnings("unchecked")
    public void copy_values_from(ScSolvingInputConf prev) {
        values = new Vector[prev.values.length];
        set_cnt = new Vector[prev.set_cnt.length];
        for (int a = 0; a < values.length; a++) {
            values[a] = (Vector<Integer>) prev.values[a].clone();
            set_cnt[a] = (Vector<Integer>) prev.set_cnt[a].clone();
        }
        untilv = prev.untilv.clone();
        next = prev.next.clone();
        num_uids = prev.num_uids;
    }

    @SuppressWarnings("unchecked")
    private void realloc(int min_length) {
        DebugOut.print_mt("realloc");
        int next_length = Math.max(min_length, values.length * 2);
        Vector<Integer>[] next_values = new Vector[next_length];
        Vector<Integer>[] next_set_cnt = new Vector[next_length];
        int[] next_untilv = new int[next_length];
        int[] next_next = new int[next_length];
        System.arraycopy(values, 0, next_values, 0, values.length);
        System.arraycopy(set_cnt, 0, next_set_cnt, 0, set_cnt.length);
        System.arraycopy(untilv, 0, next_untilv, 0, untilv.length);
        System.arraycopy(next, 0, next_next, 0, next.length);
        for (int a = values.length; a < next_length; a++) {
            next_values[a] = new Vector<Integer>(10);
            next_set_cnt[a] = new Vector<Integer>(10);
            next_untilv[a] = -1;
        }
        values = next_values;
        set_cnt = next_set_cnt;
        untilv = next_untilv;
        next = next_next;
    }

    @Override
    public int dynamicNextValue(int uid, int untilv) {
        if (uid >= values.length) {
            realloc(uid + 1);
            num_uids = uid + 1;
        }
        this.untilv[uid] = untilv;
        return nextValue(uid);
    }

    @Override
    public int nextValue(int uid) {
        final int uid_next = next[uid];
        if (uid_next >= values[uid].size()) {
            stack.add_entry(log_type, uid, uid_next);
            values[uid].add(0);
        }
        int v = values[uid].get(uid_next);
        next[uid] += 1;
        return v;
    }

    @Override
    public Vector<ScConstructValueString> getValueString(int uid) {
        DebugOut.assertFalse("set fixed input values to get a value string.");
        return null;
    }
}
