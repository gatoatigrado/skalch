package sketch.dyn.ga.base;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.assertFalse;
import static sketch.util.ScArrayUtil.extend_arr;

import java.util.Arrays;
import java.util.LinkedList;

import sketch.util.ScCloneable;
import ec.util.MersenneTwisterFast;

/**
 * a particular configuration, mapped into the (type, uid, subuid) semantics by
 * the phenotype entry.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGenotype implements ScCloneable<ScGenotype> {
    public int[] data = new int[0];
    public boolean[] active_data = new boolean[0];
    public float[] mutate_prob = new float[0];
    public LinkedList<Integer> mutate_list = new LinkedList<Integer>();

    @Override
    public String toString() {
        return "ScGenotype [active_data=" + Arrays.toString(active_data)
                + ", data=" + Arrays.toString(data) + "]";
    }

    public void reset_accessed() {
        for (int a = 0; a < active_data.length; a++) {
            active_data[a] = false;
        }
    }

    public int getValue(int idx, int untilv) {
        if (idx >= data.length) {
            realloc(2 * idx + 1);
        }
        int result = data[idx];
        active_data[idx] = true;
        if (mutate_prob[idx] == 0.f) {
            mutate_prob[idx] = 0.1f;
            mutate_list.add(idx);
        }
        if (result >= untilv) {
            result = data[idx] % untilv;
            data[idx] = result;
        }
        return result;
    }

    /** does not clone; changes this object's values */
    public boolean mutate() {
        // print_colored(BASH_GREY, "[ga]", " ", false, "mutate");
        MersenneTwisterFast local_mt = mt();
        boolean mutated = false;
        while (!mutated) {
            if (mutate_list.size() == 0) {
                assertFalse("no constructs with significant mutate probability");
            }
            for (Integer idx : mutate_list) {
                if (local_mt.nextFloat() < mutate_prob[idx]) {
                    data[idx] = Math.abs(local_mt.nextInt());
                    mutated = true;
                }
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public ScGenotype clone() {
        ScGenotype result = new ScGenotype();
        result.data = data.clone();
        result.active_data = active_data.clone();
        result.mutate_prob = mutate_prob.clone();
        result.mutate_list = (LinkedList<Integer>) mutate_list.clone();
        return result;
    }

    /** reallocate and set initial random data */
    private void realloc(int length) {
        int prev_length = data.length;
        data = extend_arr(data, length);
        active_data = extend_arr(active_data, length);
        mutate_prob = extend_arr(mutate_prob, length);
        MersenneTwisterFast mt_local = mt();
        for (int a = prev_length; a < length; a++) {
            data[a] = Math.abs(mt_local.nextInt());
        }
    }

    private int mutate_different(int change_idx, ScGenotype other,
            MersenneTwisterFast mt_local)
    {
        for (; change_idx < other.data.length; change_idx += 1) {
            if (active_data[change_idx]
                    && (data[change_idx] != other.data[change_idx]))
            {
                // print_colored(BASH_GREY, "[ga]", " ", false, "mutate index",
                // change_idx);
                data[change_idx] = Math.abs(mt_local.nextInt());
                return change_idx + 1;
            }
        }
        return -1;
    }

    public void crossover(ScGenotype other, float prob_mutate_different) {
        if (other.data.length > data.length) {
            realloc(other.data.length);
        }
        MersenneTwisterFast mt_local = mt();
        if (mt_local.nextFloat() < prob_mutate_different) {
            // print_colored(BASH_GREY, "[ga]", " ", false, "crossover mutate");
            int change_idx = 0;
            while (change_idx != -1) {
                // find next differing, and set to a random value
                change_idx = mutate_different(change_idx, other, mt_local);
                if (mt_local.nextFloat() >= prob_mutate_different) {
                    return;
                }
            }
        } else {
            // print_colored(BASH_GREY, "[ga]", " ", false,
            // "one-point crossover");
            int randidx = mt_local.nextInt(other.data.length);
            System.arraycopy(other.data, randidx, data, randidx,
                    other.data.length - randidx);
        }
    }

    public String formatIndex(int idx) {
        if (idx >= data.length) {
            return "(out of bounds)";
        } else {
            return data[idx] + ", "
                    + (active_data[idx] ? "active" : "inactive");
        }
    }
}
