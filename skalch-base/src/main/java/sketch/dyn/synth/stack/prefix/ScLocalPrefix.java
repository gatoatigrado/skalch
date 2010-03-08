package sketch.dyn.synth.stack.prefix;

import sketch.util.DebugOut;

/**
 * A thread-local prefix. This prefix can represent several levels of the stack.
 * If other threads are idle, then it can be converted to a SubtreePrefix (not
 * currently implemented).
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocalPrefix extends ScPrefix {
    public int nlinksToShared;
    public ScSharedPrefix basePrefix;

    public ScLocalPrefix(int nlinksToShared, ScSharedPrefix currentPrefix) {
        this.nlinksToShared = nlinksToShared;
        basePrefix = currentPrefix;
    }

    @Override
    public boolean getAllSearched() {
        return false;
    }

    @Override
    public ScPrefix getParent(ScPrefixSearch search) {
        if (nlinksToShared < 1) {
            DebugOut.assertFalse("dummy assert");
        }
        if (nlinksToShared == 1) {
            return basePrefix;
        } else {
            nlinksToShared -= 1;
            return this;
        }
    }

    @Override
    public void setAllSearched() {
    }

    @Override
    public ScPrefix addEntries(int addedEntries) {
        nlinksToShared += addedEntries;
        return this;
    }

    @Override
    public int nextValue() {
        DebugOut.assertFalse("don't call next value with local searches!");
        return 0;
    }
}
