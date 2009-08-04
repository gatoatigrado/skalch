package sketch.dyn.stack.prefix;

/**
 * represents the beginning of a stack. currently for synthesis only, but those
 * specifics should be kept to a minimum.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScPrefix {
    /**
     * Since a prefix doesn't know about untilv, it might go over it. If this
     * happens, the caller should call set_all_searched().
     */
    public abstract int next_value();

    /** have all of the direct descendants of this subtree been searched? */
    public abstract boolean get_all_searched();

    public abstract void set_all_searched();

    /**
     * get the parent prefix to continue the search.
     * @param stack
     *            The current search stack. Set null if there is none.
     */
    public abstract ScPrefix get_parent(ScPrefixSearch search);

    /**
     * most prefix searches are implicit through operations re-done on a local
     * stack instance, but this is useful to allow new or done threads to help.
     */
    public ScPrefixSearch explicit_prefix = null;

    public abstract ScPrefix add_entries(int added_entries);
}
