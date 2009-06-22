package sketch.dyn.inputs;

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
        return "Oracle[" + info.uid() + "] =" + values.toString() + ", "
                + values.capacity() + values.size();
    }

    public ScFixedInputGenerator createFixedInput() {
        int[] result_values = new int[values.size()];
        for (int a = 0; a < result_values.length; a++) {
            result_values[a] = values.get(a);
        }
        return new ScFixedInputGenerator(result_values, overflow_oblivious);
    }

    @Override
    public int next_value() {
        if (next >= values.size()) {
            // add element, might fail (in which case, don't add the value yet)
            al.add_entry(log_type, info.uid(), next);
            values.add(0);
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

    /**
     * reset_index should happen every _run_ but not between _counterexamples_.
     * it is currently called in the end of ScStack.next()
     */
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

    /** called after the stack is popped */
    public void reset_accessed(int subuid) {
        // TODO - remove asserts and next -= 1
        if (next != values.size()) {
            DebugOut.assertFalse("i think this is correct...", next, values
                    .size());
        }
        next -= 1; // N.B. - for asserts ONLY!!! next := 0 from reset_index()
        if (subuid != next) {
            DebugOut.assertFalse(
                    "ScSolvingInputGenerator - bad value to pop from stack",
                    subuid, next);
        }

        values.remove(subuid);
    }
}
