package sketch.dyn.ga;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.ScArrayUtil.extend_arr;
import sketch.util.DebugOut;
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

    public int getValue(int idx, int untilv) {
        if (idx > data.length) {
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
        MersenneTwisterFast local_mt = mt();
        for (int a = 0; a < 16; a++) {
            int idx = local_mt.nextInt(data.length);
            data[idx] = local_mt.nextInt();
            if (active_data[idx]) {
                return;
            }
        }
        DebugOut.print_mt("didn't mutate anything; "
                + "consider searching for longer.");
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
            data[a] = mt_local.nextInt();
        }
    }

    public void crossover(ScGenotype other) {
        if (other.data.length > data.length) {
            realloc(other.data.length);
        }
        int randidx = mt().nextInt(other.data.length);
        System.arraycopy(other.data, randidx, data, randidx, other.data.length
                - randidx);
    }
}
