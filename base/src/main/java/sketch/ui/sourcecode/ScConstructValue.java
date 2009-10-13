package sketch.ui.sourcecode;

import sketch.util.DebugOut;

/**
 * keep this generic in the case that we will someday get float support
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScConstructValue {
    public Object value;

    public ScConstructValue(int v) {
        value = v;
    }

    public ScConstructValue(float v) {
        value = v;
    }

    public String formatString() {
        return value.toString();
    }

    public int intValue() {
        if (value.getClass() != Integer.class) {
            DebugOut.assertFalse("bad call to intValue()");
        }
        return ((Integer) value).intValue();
    }
}
