package sketch.dyn.synth.stack.prefix;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import sketch.util.DebugOut;
import sketch.util.MTSafe;

/**
 * A prefix which may be shared by many threads; thus, synchronization is
 * necessary.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
@MTSafe
public class ScSharedPrefix extends ScPrefix {
    public AtomicBoolean search_done = new AtomicBoolean(false);
    public AtomicInteger next_value = new AtomicInteger(0);
    public int nparentlinks;
    public AtomicReference<ScPrefix> parent =
            new AtomicReference<ScPrefix>(null);

    public ScSharedPrefix(int nparentlinks) {
        this.nparentlinks = nparentlinks;
    }

    @Override
    public String toString() {
        return "SharedPrefix[p-" + nparentlinks + "(parent)]";
    }

    @Override
    public boolean get_all_searched() {
        // return false;
        return search_done.get();
    }

    @Override
    public int next_value() {
        return next_value.incrementAndGet();
    }

    @Override
    public synchronized ScPrefix get_parent(ScPrefixSearch search) {
        ScPrefix curr_parent = parent.get();
        if (curr_parent == null) {
            curr_parent = new ScSharedPrefix(nparentlinks + 1);
            if (!parent.compareAndSet(null, curr_parent)) {
                curr_parent = parent.get();
            }
        }
        return curr_parent;
    }

    @Override
    public void set_all_searched() {
        search_done.set(true);
    }

    @Override
    public ScPrefix add_entries(int added_entries) {
        if (added_entries <= 0) {
            DebugOut.assertFalse("scsharedprefix - added entries is zero");
        }
        // System.identityHashCode();
        return new ScLocalPrefix(added_entries, this);
    }
}
