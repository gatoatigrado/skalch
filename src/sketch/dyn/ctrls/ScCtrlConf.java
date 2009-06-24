package sketch.dyn.ctrls;

import java.util.Arrays;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.util.Vector4;

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
            ssr_holes[hole_info[a].uid()] =
                    new ScSSRHoleValue(al, log_type, hole_info[a]);
            fixed_holes[a] = new ScFixedHoleValue();
        }
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (ScSSRHoleValue ssr_hole : ssr_holes) {
            result *= 3333;
            result += ssr_hole.v;
        }
        return result;
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

    /** fixed holes with valueString set */
    public ScHoleValue[] fixed_annotated_controls() {
        int[] set_cnt = new int[ssr_holes.length];
        for (int a = 0; a < ssr_holes.length; a++) {
            set_cnt[a] = ssr_holes[a].set_cnt;
        }
        Arrays.sort(set_cnt);
        // distribute colors evenly, then go halfway towards closer neighbor.
        for (int a = 0; a < ssr_holes.length; a++) {
            fixed_holes[a].v = ssr_holes[a].v;
            float c = (a) / (2.f * (ssr_holes.length - 1));
            c += ssr_holes[a].set_cnt / (2.f * set_cnt[set_cnt.length - 1]);
            fixed_holes[a].myValueString =
                    gen_value_string(c, fixed_holes[a].v);
        }
        return fixed_holes;
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
        return "<span style=\"color: " + color + ";\">" + v + "</span>";
    }

    public void reset_accessed(int uid) {
        ssr_holes[uid].reset_accessed();
    }

    public void copy_from(ScCtrlConf ctrls) {
        for (int a = 0; a < ctrls.ssr_holes.length; a++) {
            ssr_holes[a].accessed = ctrls.ssr_holes[a].accessed;
            ssr_holes[a].v = ctrls.ssr_holes[a].v;
            ssr_holes[a].set_cnt = ctrls.ssr_holes[a].set_cnt;
        }
    }
}
