package sketch.dyn.inputs;

import sketch.util.DebugOut;

/**
 * Analogy of ScFixedHoleValue; perhaps faster (array versus vector), and
 * doesn't log accesses
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScFixedInputGenerator extends ScInputGenerator {
    protected int[] values;
    protected int next = 0;
    public boolean overflow_oblivious;

    public ScFixedInputGenerator(int[] values, boolean overflow_oblivious) {
        this.values = values.clone();
        this.overflow_oblivious = overflow_oblivious;
    }

    public void reset_index() {
        next = 0;
    }

    @Override
    public int next_value() {
        if (next >= values.length) {
            DebugOut.assert_(overflow_oblivious,
                    "fixed input generator exceeding length", values.length);
            DebugOut.print("fixed input generator exceeding length...", next,
                    values.length);
            return 0;
        } else {
            int rv = values[next];
            next += 1;
            return rv;
        }
    }
}
