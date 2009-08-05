package sketch.dyn.synth.stack.prefix;

/**
 * A specific search, with a position in the search space. Must correctly
 * implement Cloneable.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScPrefixSearch implements Cloneable {
    public ScPrefix current_prefix;

    public abstract ScPrefixSearch clone();
}
