package sketch.dyn.constructs.inputs;

import static sketch.util.fcns.ScArrayUtil.extend_arr;

import java.util.Vector;

import sketch.dyn.synth.stack.ScStack;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.util.DebugOut;
import sketch.util.wrapper.ScRichString;

/**
 * Common functions on an array of solving inputs. These can be converted to fixed inputs.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScSolvingInputConf extends ScInputConf implements Cloneable {
    protected ScStack stack;
    protected int logType;
    protected Vector<Integer>[] values;
    public Vector<Integer>[] untilv;
    protected Vector<Integer>[] setCnt;
    protected int[] defaultUntilv;
    protected int[] next;
    protected int numUids = 0;

    @SuppressWarnings("unchecked")
    public ScSolvingInputConf(ScStack stack, int logType) {
        this.stack = stack;
        this.logType = logType;
        values = new Vector[0]; // can't create generic array
        setCnt = new Vector[0];
        untilv = new Vector[0]; // can't create generic array
        defaultUntilv = new int[0];
        next = new int[0];
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

    @Override
    public String toString() {
        return "ScSolvingInputConf[ values=" + (new ScRichString(", ")).join(values) +
                ",\nuntilv=" + (new ScRichString(", ")).join(untilv) + " ]";
    }

    /** for clone method only */
    protected ScSolvingInputConf() {}

    /** for ScTestGenerator */
    public void addInput(int uid, int v) {
        if (uid >= values.length) {
            realloc(uid + 1);
        }
        values[uid].add(v);
    }

    /** copies synthesized inputs to fixed inputs. result is threadsafe / unique */
    public ScFixedInputConf fixedInputs() {
        int[][] fixedValues = new int[numUids][];
        int[][] setCntValues = new int[numUids][];
        for (int a = 0; a < numUids; a++) {
            fixedValues[a] = new int[values[a].size()];
            setCntValues[a] = new int[values[a].size()];
            for (int c = 0; c < fixedValues[a].length; c++) {
                fixedValues[a][c] = values[a].get(c);
                if (setCnt[a].size() <= c || setCnt[a].get(c) == null) {
                    setCntValues[a][c] = 0;
                } else {
                    setCntValues[a][c] = setCnt[a].get(c);
                }
            }
        }
        // defaultUntilv is only used for the inputs, not the oracles
        // FIXME - make it correct for both.
        return new ScFixedInputConf(fixedValues, setCntValues, defaultUntilv.clone(),
                next.clone());
    }

    public boolean set(int uid, int subuid, int v) {
        if (v < untilv[uid].get(subuid)) {
            if (setCnt[uid].size() <= subuid) {
                setCnt[uid].setSize(subuid + 1);
            }
            final Integer prev = setCnt[uid].get(subuid);
            if (prev == null) {
                setCnt[uid].set(subuid, 1);
            } else {
                setCnt[uid].set(subuid, prev + 1);
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

    public void resetIndex() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }

    public void resetAccessed(int uid, int subuid) {
        values[uid].remove(subuid);
        untilv[uid].remove(subuid);
    }

    @SuppressWarnings("unchecked")
    public void copyValuesFrom(ScSolvingInputConf prev) {
        values = new Vector[prev.values.length];
        untilv = new Vector[prev.untilv.length];
        setCnt = new Vector[prev.setCnt.length];
        for (int a = 0; a < values.length; a++) {
            values[a] = (Vector<Integer>) prev.values[a].clone();
            untilv[a] = (Vector<Integer>) prev.untilv[a].clone();
            setCnt[a] = (Vector<Integer>) prev.setCnt[a].clone();
        }
        next = prev.next.clone();
        numUids = prev.numUids;
    }

    private void realloc(int minLength) {
        DebugOut.print_mt("realloc");
        int nextLength = Math.max(minLength, values.length * 2);
        values = extend_arr(values, nextLength, 10, 0);
        setCnt = extend_arr(setCnt, nextLength, 10, 0);
        untilv = extend_arr(untilv, nextLength, 10, 0);
        next = extend_arr(next, nextLength);
    }

    @Override
    public int dynamicNextValue(int uid, int untilv_) {
        if (uid >= values.length) {
            realloc(uid + 1);
        }
        if (uid >= numUids) {
            numUids = uid + 1;
        }
        return nextValueWithUntilv(uid, untilv_);
    }

    @Override
    public int nextValue(int uid) {
        return nextValueWithUntilv(uid, defaultUntilv[uid]);
    }

    private int nextValueWithUntilv(int uid, int untilv_) {
        final int uidNext = next[uid];
        if (uidNext >= values[uid].size()) {
            stack.addEntry(logType, uid, uidNext);
            values[uid].add(0);
            untilv[uid].add(untilv_);
        }
        int v = values[uid].get(uidNext);
        next[uid] += 1;
        return v;
    }

    @Override
    public Vector<ScConstructValueString> getValueString(int uid) {
        DebugOut.assertFalse("set fixed input values to get a value string.");
        return null;
    }

    @Override
    public int[] getValueArray() {
        Vector<Integer> linearized = new Vector<Integer>();
        for (Vector<Integer> constructValues : values) {
            linearized.addAll(constructValues);
        }
        Integer[] boxed = linearized.toArray(new Integer[0]);
        int[] result = new int[boxed.length];
        for (int a = 0; a < boxed.length; a++) {
            result[a] = boxed[a];
        }
        return result;
    }
}
