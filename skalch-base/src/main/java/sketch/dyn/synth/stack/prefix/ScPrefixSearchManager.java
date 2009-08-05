package sketch.dyn.synth.stack.prefix;

import sketch.util.DebugOut;
import sketch.util.FilteredQueue;

/**
 * Manages searches of subtrees. Currently doesn't do much.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPrefixSearchManager<SearchType extends ScPrefixSearch> {
    protected SearchType search_default;
    protected ScDefaultPrefix prefix_default;
    protected ActivePrefixQueue active_prefixes = new ActivePrefixQueue();

    public ScPrefixSearchManager(SearchType search_default,
            ScDefaultPrefix prefix_default)
    {
        this.search_default = search_default;
        this.prefix_default = prefix_default;
    }

    @SuppressWarnings("unchecked")
    public SearchType clone_default_search() {
        return (SearchType) search_default.clone();
    }

    public void add_prefix(ScSubtreePrefix prefix) {
        DebugOut.not_implemented("add_prefix");
        if (prefix.explicit_prefix == null) {
            DebugOut.assertFalse("must have explicit prefix for cloning");
        }
        active_prefixes.add(prefix);
    }

    @SuppressWarnings("unchecked")
    public SearchType get_active_prefix() {
        DebugOut.not_implemented("get_active_prefix");
        return (SearchType) active_prefixes.get().explicit_prefix.clone();
    }

    public class ActivePrefixQueue extends FilteredQueue<ScSharedPrefix> {
        @Override
        public boolean apply_filter(ScSharedPrefix elt) {
            DebugOut.not_implemented("apply filter...");
            return !elt.get_all_searched();
        }
    }
}
