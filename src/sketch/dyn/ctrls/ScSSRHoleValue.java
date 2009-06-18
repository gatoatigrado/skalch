package sketch.dyn.ctrls;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.util.DebugOut;

/**
 * The element the backend API will call when resolving the value of a hole
 * during synthesis.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public final class ScSSRHoleValue extends ScHoleValue {
    protected ScStack al;
    protected int log_type;
    protected ScConstructInfo info;

    public int v;
    public int untilv;
    public boolean accessed;

    public ScSSRHoleValue(ScStack al, int log_type, ScConstructInfo info) {
        this.al = al;
        this.log_type = log_type;
        this.info = info;
        this.untilv = info.untilv();
    }

    public boolean set(int v) {
        if (v < untilv) {
            this.v = v;
            return true;
        } else {
            // reset for next stack iteration
            this.v = 0;
            return false;
        }
    }

    @Override
    public int get_value() {
        if (!accessed) {
            al.add_entry(log_type, info.uid(), 0);
        }
        accessed = true;
        return v;
    }

    public void reset_accessed() {
        DebugOut.assert_(accessed,
                "ScSSRHoleValue - bad value to pop from stack");
        accessed = false;
    }
}
