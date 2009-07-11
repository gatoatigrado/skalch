package sketch.dyn.ga.base;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.BASH_GREY;
import static sketch.util.DebugOut.print_colored;
import static sketch.util.ScArrayUtil.extend_arr;

import java.util.Arrays;

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
        if (result >= untilv) {
            result = data[idx] % untilv;
            data[idx] = result;
        }
        return result;
    }

    /** does not clone; changes this object's values */
    public void mutate() {
        print_colored(BASH_GREY, "[ga]", " ", false, "mutate");
        MersenneTwisterFast local_mt = mt();
        for (int a = 0; a < 16; a++) {
            int idx = local_mt.nextInt(data.length);
            data[idx] = Math.abs(local_mt.nextInt());
            if (active_data[idx]) {
                return;
            }
        }
        // DebugOut.print_mt("didn't mutate anything; "
        // + "consider searching for longer.");
    }

    @Override
    public ScGenotype clone() {
        ScGenotype result = new ScGenotype();
        result.data = data.clone();
        result.active_data = active_data.clone();
        return result;
    }

    /** reallocate and set initial random data */
    private void realloc(int length) {
        int prev_length = data.length;
        data = extend_arr(data, length);
        active_data = extend_arr(active_data, length);
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
                print_colored(BASH_GREY, "[ga]", " ", false, "mutate index",
                        change_idx);
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
            print_colored(BASH_GREY, "[ga]", " ", false, "crossover mutate");
            int change_idx = 0;
            while (change_idx != -1) {
                // find next differing, and set to a random value
                change_idx = mutate_different(change_idx, other, mt_local);
                if (mt_local.nextFloat() >= prob_mutate_different) {
                    return;
                }
            }
        } else {
            print_colored(BASH_GREY, "[ga]", " ", false, "one-point crossover");
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
