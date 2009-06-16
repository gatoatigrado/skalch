package sketch.dyn.inputs;

/**
 * An array of ScInputGenerator is used by the Scala API when resolving input
 * values.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScInputGenerator {
    public abstract int next_value();
}
