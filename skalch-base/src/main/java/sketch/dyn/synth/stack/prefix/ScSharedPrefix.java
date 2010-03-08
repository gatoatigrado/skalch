package sketch.dyn.synth.stack.prefix;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import sketch.util.DebugOut;
import sketch.util.thread.MTSafe;

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
    public AtomicBoolean searchDone = new AtomicBoolean(false);
    public AtomicInteger nextValue = new AtomicInteger(0);
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
    public boolean getAllSearched() {
        // return false;
        return searchDone.get();
    }

    @Override
    public int nextValue() {
        return nextValue.incrementAndGet();
    }

    @Override
    public synchronized ScPrefix getParent(ScPrefixSearch search) {
        ScPrefix currParent = parent.get();
        if (currParent == null) {
            currParent = new ScSharedPrefix(nparentlinks + 1);
            if (!parent.compareAndSet(null, currParent)) {
                currParent = parent.get();
            }
        }
        return currParent;
    }

    @Override
    public void setAllSearched() {
        searchDone.set(true);
    }

    @Override
    public ScPrefix addEntries(int addedEntries) {
        if (addedEntries <= 0) {
            DebugOut.assertFalse("scsharedprefix - added entries is zero");
        }
        // System.identityHashCode();
        return new ScLocalPrefix(addedEntries, this);
    }
}
