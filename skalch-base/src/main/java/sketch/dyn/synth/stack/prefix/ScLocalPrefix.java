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
    public int nlinks_to_shared;
    public ScSharedPrefix base_prefix;

    public ScLocalPrefix(int nlinks_to_shared, ScSharedPrefix current_prefix) {
        this.nlinks_to_shared = nlinks_to_shared;
        base_prefix = current_prefix;
    }

    @Override
    public boolean get_all_searched() {
        return false;
    }

    @Override
    public ScPrefix get_parent(ScPrefixSearch search) {
        if (nlinks_to_shared < 1) {
            DebugOut.assertFalse("dummy assert");
        }
        if (nlinks_to_shared == 1) {
            return base_prefix;
        } else {
            nlinks_to_shared -= 1;
            return this;
        }
    }

    @Override
    public void set_all_searched() {
    }

    @Override
    public ScPrefix add_entries(int added_entries) {
        nlinks_to_shared += added_entries;
        return this;
    }

    @Override
    public int next_value() {
        DebugOut.assertFalse("don't call next value with local searches!");
        return 0;
    }
}
