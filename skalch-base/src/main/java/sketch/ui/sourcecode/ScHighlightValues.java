package sketch.ui.sourcecode;

import java.awt.Color;
import java.util.Vector;

import sketch.util.gui.Vector4;

/**
 * Highlight an array of values, based on which ones are more recently accessed.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScHighlightValues {
    private static ScConstructValueString genValueString(float f, ScConstructValue inner)
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
                first.scalar_multiply(amntFirst).add(second.scalar_multiply(amntSecond)).add(
                        third.scalar_multiply(amntThird));
        String color = theColor.hexColor();
        return new ScConstructValueString("<span style=\"color: " + color + ";\">",
                inner, "</span>");
    }

    public static String getColorString(Color c) {
        String red = Integer.toHexString(c.getRed());
        while (red.length() < 2) {
            red = '0' + red;
        }

        String green = Integer.toHexString(c.getGreen());
        while (green.length() < 2) {
            green = '0' + green;
        }

        String blue = Integer.toHexString(c.getBlue());
        while (blue.length() < 2) {
            blue = '0' + blue;
        }

        return red + green + blue;
    }

    private static ScConstructValueString genValueString(Color c, ScConstructValue inner)
    {
        return new ScConstructValueString("<span style=\"color: " + getColorString(c) +
                ";\">", inner, "</span>");
    }

    public static void genValueStrings(Vector<Value> values) {
        if (values.size() == 0) {
            return;
        }
        for (int a = 0; a < values.size(); a++) {
            Color c = values.get(a).color;
            values.get(a).result = genValueString(c, values.get(a).value);
        }
    }

    public static class Value {
        protected ScConstructValue value;
        public ScConstructValueString result;
        protected Color color;
        protected float countValue;
        public Object tag;

        public Value(ScConstructValue value, float countValue, Object tag) {
            this.value = value;
            this.countValue = countValue;
            this.tag = tag;
            color = Color.black;
        }

        public Value(ScConstructValue value, float countValue, Object tag, Color color) {
            this.value = value;
            this.countValue = countValue;
            this.tag = tag;
            this.color = color;
        }

        public static float vectorMax(Vector<Value> values) {
            float max = values.get(0).countValue;
            for (Value v : values) {
                if (v.countValue > max) {
                    max = v.countValue;
                }
            }
            return max;
        }
    }
}
