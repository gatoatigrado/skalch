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
    public int v;
    public int uid;
    public int untilv;
    public int set_cnt = 0;
    public boolean accessed;

    public ScSSRHoleValue(ScStack al, int log_type, ScConstructInfo info) {
        this.al = al;
        this.log_type = log_type;
        uid = info.uid();
        untilv = info.untilv();
    }

    @Override
    public String toString() {
        return "Hole[" + uid + "] = " + v;
    }

    public boolean set(int v) {
        set_cnt += 1;
        if (v < untilv) {
            this.v = v;
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int get_value() {
        if (!accessed) {
            // might fail; don't set anything yet.
            al.add_entry(log_type, uid, 0);
        }
        accessed = true;
        return v;
    }

    public void reset_accessed() {
        if (!accessed) {
            DebugOut
                    .assertFalse("ScSSRHoleValue - bad value to pop from stack");
        }
        accessed = false;
        v = 0;
    }

    @Override
    public String get_value_string() {
        DebugOut
                .assertFalse("don't call getValueString on a synthesizing input.");
        return null;
    }
}
