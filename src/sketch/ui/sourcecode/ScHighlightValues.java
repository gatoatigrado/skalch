package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.util.Vector4;

/**
 * Highlight an array of values, based on which ones are more recently accessed.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScHighlightValues {
    private static String gen_value_string(float f, String inner) {
        float amnt_first, amnt_second, amnt_third;
        if (f < 0.5) {
            amnt_first = 1 - 2 * f;
            amnt_second = 2 * f;
            amnt_third = 0;
        } else {
            amnt_first = 0;
            amnt_second = 2 - 2 * f;
            amnt_third = 2 * f - 1;
        }
        Vector4 first = new Vector4(0.f, 0.f, 1.f, 1.f);
        Vector4 second = new Vector4(0.8f, 0.63f, 0.f, 1.f);
        Vector4 third = new Vector4(1.f, 0.f, 0.f, 1.f);
        Vector4 the_color =
                first.scalar_multiply(amnt_first).add(
                        second.scalar_multiply(amnt_second)).add(
                        third.scalar_multiply(amnt_third));
        String color = the_color.hexColor();
        return "<span style=\"color: " + color + ";\">" + inner + "</span>";
    }

    public static void gen_value_strings(Vector<Value> values) {
        if (values.size() == 0) {
            return;
        }
        float maxv = Value.vector_max(values);
        for (int a = 0; a < values.size(); a++) {
            float c = values.get(a).color_value / maxv;
            values.get(a).result = gen_value_string(c, values.get(a).inner);
        }
    }

    public static class Value {
        protected String inner;
        public String result;
        protected float color_value;
        protected Object tag;

        public Value(String inner, int color_value, Object tag) {
            this.inner = inner;
            this.color_value = color_value;
            this.tag = tag;
        }

        public static float vector_max(Vector<Value> values) {
            float max = values.get(0).color_value;
            for (Value v : values) {
                if (v.color_value > max) {
                    max = v.color_value;
                }
            }
            return max;
        }
    }
}
