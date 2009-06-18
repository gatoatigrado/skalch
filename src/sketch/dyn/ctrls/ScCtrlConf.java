package sketch.dyn.ctrls;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;

/**
 * Wrapper for an array of holes.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScCtrlConf {
    public ScSSRHoleValue[] ssr_holes;
    protected ScFixedHoleValue[] fixed_holes;

    public ScCtrlConf(ScConstructInfo[] hole_info, ScStack al, int log_type) {
        ssr_holes = new ScSSRHoleValue[hole_info.length];
        fixed_holes = new ScFixedHoleValue[hole_info.length];
        for (int a = 0; a < hole_info.length; a++) {
            ssr_holes[hole_info[a].uid()] = new ScSSRHoleValue(al, log_type,
                    hole_info[a]);
            fixed_holes[a] = new ScFixedHoleValue();
        }
    }

    public boolean set(int uid, int v) {
        return ssr_holes[uid].set(v);
    }

    public int get(int uid) {
        return ssr_holes[uid].v;
    }

    /**
     * copies synthesized holes to fixed hole array and returns that. result is
     * not threadsafe / unique
     */
    public ScFixedHoleValue[] fixed_controls() {
        for (int a = 0; a < ssr_holes.length; a++) {
            fixed_holes[a].v = ssr_holes[a].v;
        }
        return fixed_holes;
    }

    public void reset_accessed(int uid) {
        ssr_holes[uid].reset_accessed();
    }
}
