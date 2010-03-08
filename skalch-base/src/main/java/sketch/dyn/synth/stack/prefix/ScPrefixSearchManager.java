package sketch.dyn.synth.stack.prefix;

import sketch.util.DebugOut;
import sketch.util.datastructures.FilteredQueue;

/**
 * Manages searches of subtrees. Currently doesn't do much.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScPrefixSearchManager<SearchType extends ScPrefixSearch> {
    protected SearchType searchDefault;
    protected ScDefaultPrefix prefixDefault;
    protected ActivePrefixQueue activePrefixes = new ActivePrefixQueue();

    public ScPrefixSearchManager(SearchType searchDefault, ScDefaultPrefix prefixDefault)
    {
        this.searchDefault = searchDefault;
        this.prefixDefault = prefixDefault;
    }

    @SuppressWarnings("unchecked")
    public SearchType cloneDefaultSearch() {
        return (SearchType) searchDefault.clone();
    }

    public void addPrefix(ScSubtreePrefix prefix) {
        DebugOut.not_implemented("addPrefix");
        if (prefix.explicitPrefix == null) {
            DebugOut.assertFalse("must have explicit prefix for cloning");
        }
        activePrefixes.add(prefix);
    }

    @SuppressWarnings("unchecked")
    public SearchType getActivePrefix() {
        DebugOut.not_implemented("getActivePrefix");
        return (SearchType) activePrefixes.get().explicitPrefix.clone();
    }

    public class ActivePrefixQueue extends FilteredQueue<ScSharedPrefix> {
        @Override
        public boolean apply_filter(ScSharedPrefix elt) {
            DebugOut.not_implemented("apply filter...");
            return !elt.getAllSearched();
        }
    }
}
