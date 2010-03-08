package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.util.gui.Vector4;

/**
 * Highlight an array of values, based on which ones are more recently accessed.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScHighlightValues {
    private static ScConstructValueString genValueString(float f,
            ScConstructValue inner)
    {
        float amntFirst, amntSecond, amntThird;
        if (f < 0.5) {
            amntFirst = 1 - 2 * f;
            amntSecond = 2 * f;
            amntThird = 0;
        } else {
            amntFirst = 0;
            amntSecond = 2 - 2 * f;
            amntThird = 2 * f - 1;
        }
        Vector4 first = new Vector4(0.f, 0.f, 1.f, 1.f);
        Vector4 second = new Vector4(0.8f, 0.63f, 0.f, 1.f);
        Vector4 third = new Vector4(1.f, 0.f, 0.f, 1.f);
        Vector4 theColor =
                first.scalar_multiply(amntFirst).add(
                        second.scalar_multiply(amntSecond)).add(
                        third.scalar_multiply(amntThird));
        String color = theColor.hexColor();
        return new ScConstructValueString("<span style=\"color: " + color
                + ";\">", inner, "</span>");
    }

    public static void genValueStrings(Vector<Value> values) {
        if (values.size() == 0) {
            return;
        }
        float maxv = Value.vectorMax(values);
        for (int a = 0; a < values.size(); a++) {
            float c = values.get(a).colorValue / maxv;
            values.get(a).result = genValueString(c, values.get(a).value);
        }
    }

    public static class Value {
        protected ScConstructValue value;
        public ScConstructValueString result;
        protected float colorValue;
        public Object tag;

        public Value(ScConstructValue value, float colorValue, Object tag) {
            this.value = value;
            this.colorValue = colorValue;
            this.tag = tag;
        }

        public static float vectorMax(Vector<Value> values) {
            float max = values.get(0).colorValue;
            for (Value v : values) {
                if (v.colorValue > max) {
                    max = v.colorValue;
                }
            }
            return max;
        }
    }
}
