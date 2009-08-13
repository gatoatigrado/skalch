package sketch.dyn.synth.ga.base;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.assertSlow;
import static sketch.util.DebugOut.print;
import static sketch.util.fcns.ScArrayUtil.extend_arr;

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
    public float[] _mutate_prob = new float[0];
    public float total_mutate_prob = 0;

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

    public void change_mutate_prob(int idx, float value) {
        float prev = _mutate_prob[idx];
        _mutate_prob[idx] = value;
        total_mutate_prob += value - prev;
    }

    /** ensure that mutate() doesn't loop unnecessarily if prob. are low */
    public void readjust_mutate_prob() {
        if (total_mutate_prob <= 0) {
            print("total mutate probility is zero");
            return;
        }
        if (total_mutate_prob < 0.5) {
            for (int idx = 0; idx < _mutate_prob.length; idx++) {
                change_mutate_prob(idx, _mutate_prob[idx] * 2);
            }
        }
    }

    public int getValue(int idx, int untilv) {
        if (idx >= data.length) {
            realloc(2 * idx + 1);
        }
        int result = data[idx];
        active_data[idx] = true;
        if (_mutate_prob[idx] == 0.f) {
            change_mutate_prob(idx, 0.1f);
        }
        if (result >= untilv) {
            result = data[idx] % untilv;
            data[idx] = result;
        }
        return result;
    }

    /** does not clone; changes this object's values */
    public void mutate() {
        // print_colored(BASH_GREY, "[ga]", " ", false, "mutate");
        MersenneTwisterFast local_mt = mt();
        readjust_mutate_prob();
        boolean mutated = false;
        while (!mutated) {
            for (int idx = 0; idx < _mutate_prob.length; idx++) {
                if (local_mt.nextFloat() < _mutate_prob[idx]) {
                    data[idx] = Math.abs(local_mt.nextInt());
                    mutated = true;
                    // return;
                }
            }
        }
    }

    @Override
    public ScGenotype clone() {
        ScGenotype result = new ScGenotype();
        result.data = data.clone();
        result.active_data = active_data.clone();
        result._mutate_prob = _mutate_prob.clone();
        result.total_mutate_prob = total_mutate_prob;
        return result;
    }

    /** reallocate and set initial random data */
    private void realloc(int length) {
        int prev_length = data.length;
        data = extend_arr(data, length);
        active_data = extend_arr(active_data, length);
        _mutate_prob = extend_arr(_mutate_prob, length);
        MersenneTwisterFast mt_local = mt();
        for (int a = prev_length; a < length; a++) {
            data[a] = Math.abs(mt_local.nextInt());
        }
    }

    public void change_probability(int idx, float next) {
        if (next > 0.01f && next < 0.5f) {
            change_mutate_prob(idx, next);
        }
    }

    public boolean crossover(ScGenotype other) {
        if (other.data.length > data.length) {
            realloc(other.data.length);
        }
        MersenneTwisterFast mt_local = mt();
        int randidx = mt_local.nextInt(other.data.length);
        boolean modified = false;
        // modify value at randidx
        {
            int min = data[randidx];
            int max = other.data[randidx];
            if (max > min) {
                min = other.data[randidx];
                max = data[randidx];
            }
            int difference = max - min;
            if (difference > 0) {
                int next =
                        Math.max(0, min - difference)
                                + mt_local.nextInt(3 * difference);
                assertSlow(next >= 0);
                data[randidx] = next;
            }
        }
        // decrease probability of mutation for values that are the same
        for (int a = 0; a < randidx + 1; a++) {
            if (data[a] == other.data[a]) {
                change_probability(a, _mutate_prob[a] * 0.9f);
            } else {
                change_probability(a, _mutate_prob[a] * 1.11111111111f);
            }
        }
        for (int a = randidx + 1; a < other.data.length; a++) {
            int other_value = other.data[a];
            if (data[a] == other_value) {
                change_probability(a, other._mutate_prob[a] * 0.9f);
            } else {
                data[a] = other_value;
                change_probability(a, other._mutate_prob[a] * 1.111111111f);
            }
        }
        return modified;
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
