package sketch.dyn.inputs;

import java.util.Stack;
import java.util.Vector;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.util.DebugOut;

/**
 * Backend called by Scala API when resolving the value of an input generator.
 * Input generators could be inputs or oracles.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public final class ScSolvingInputGenerator extends ScInputGenerator {
    protected ScStack al;
    protected int log_type;
    protected ScConstructInfo info;
    public boolean overflow_oblivious = false;

    public Vector<Integer> values = new Vector<Integer>(10);
    public int untilv;
    int next = 0; // index which will increment as the sketch is run

    public ScSolvingInputGenerator(ScStack al, int log_type,
            ScConstructInfo info)
    {
        this.al = al;
        this.log_type = log_type;
        this.info = info;
        this.untilv = info.untilv();
    }

    @Override
    public String toString() {
        return info.uid() + "=" + values.toString();
    }

    public ScFixedInputGenerator createFixedInput() {
        int[] result_values = new int[values.size()];
        for (int a = 0; a < result_values.length; a++) {
            result_values[a] = values.get(a);
        }
        return new ScFixedInputGenerator(result_values, overflow_oblivious);
    }

    public ScSolvingInputGenerator clone() {
        ScSolvingInputGenerator result = new ScSolvingInputGenerator(al,
                log_type, info);
        result.values.addAll(values);
        return result;
    }

    @Override
    public int next_value() {
        if (next >= values.size()) {
            // add element
            values.add(0);
            al.add_entry(log_type, info.uid(), next);
        }
        int v = values.get(next);
        next += 1;
        return v;
    }

    public boolean set(int subuid, int v) {
        if (v < untilv) {
            this.values.set(subuid, v);
            return true;
        } else {
            return false;
        }
    }

    public void reset_index() {
        next = 0;
    }

    public void addInput(int v) {
        if (v >= untilv) {
            DebugOut.print("warning - overflowing construct with uid", info
                    .uid(), "; declared limit", untilv);
        }
        values.add(v);
        overflow_oblivious = false;
    }

    public void reset_accessed(int subuid) {
        next -= 1;
        DebugOut.assert_(subuid == next,
                "ScSolvingInputGenerator - bad value to pop from stack");
    }
}
