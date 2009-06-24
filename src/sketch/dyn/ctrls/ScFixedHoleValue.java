package sketch.dyn.ctrls;

/**
 * A fixed hole value, after the sketch has been synthesized.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public final class ScFixedHoleValue extends ScHoleValue {
    public int v = 0;
    public String myValueString = null;

    @Override
    public int get_value() {
        return v;
    }

    @Override
    public String get_value_string() {
        if (myValueString == null) {
            return "value(" + String.valueOf(v) + ")";
        } else {
            return myValueString;
        }
    }
}
