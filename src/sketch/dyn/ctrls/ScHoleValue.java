package sketch.dyn.ctrls;

/**
 * The element the backend API will call when resolving the value of a hole.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScHoleValue {
    public abstract int get_value();

    public abstract String get_value_string();
}
