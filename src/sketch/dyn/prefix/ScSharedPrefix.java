package sketch.dyn.prefix;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import sketch.util.DebugOut;

/**
 * A prefix which may be shared by many threads; thus, synchronization is
 * necessary.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSharedPrefix extends ScPrefix {
    public AtomicBoolean search_done = new AtomicBoolean(false);
    public AtomicInteger next_value = new AtomicInteger(0);
    public AtomicReference<ScPrefix> parent = new AtomicReference<ScPrefix>(
            null);

    @Override
    public boolean get_all_searched() {
        return search_done.get();
    }

    @Override
    public int next_value(ScPrefixSearch search) {
        return next_value.incrementAndGet();
    }

    @Override
    public ScPrefix get_parent(ScPrefixSearch search) {
        ScPrefix curr_parent = parent.get();
        if (curr_parent == null) {
            curr_parent = new ScSharedPrefix();
            if (!parent.compareAndSet(null, curr_parent)) {
                curr_parent = parent.get();
            }
        }
        return curr_parent;
    }

    @Override
    public void set_all_searched() {
        // TODO Auto-generated method stub
        search_done.set(true);
    }

    @Override
    public ScPrefix add_entries(int added_entries) {
        DebugOut.assert_(added_entries > 0,
                "scsharedprefix - added entries is zero");
        return new ScLocalPrefix(added_entries, this);
    }
}
