/**
 * 
 */
package sketch.dyn.stats;

import java.util.concurrent.atomic.AtomicLong;

import sketch.util.wrapper.ScRichString;

/**
 * The base field type for ScStatsMT; it uses an atomic long counter.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatEntry {
    protected AtomicLong ctr = new AtomicLong();
    public Float value;
    public String name;
    public String short_name;

    /**
     * @param short_name
     *            name used when printing rate string
     */
    public ScStatEntry(String name, String short_name) {
        this.name = name;
        this.short_name = short_name;
    }

    @Override
    public String toString() {
        return formatString(0);
    }

    public String formatString(int align) {
        return (new ScRichString(name)).lpad(align) + ": "
                + String.format("%9.1f", value);
    }

    public float rate(ScStatEntry base) {
        if (value == 0) {
            return 0;
        }
        return value / base.value;
    }

    public ScStatEntry(String name) {
        this(name, name);
    }

    public float get_value() {
        value = new Float(ctr.get());
        return value;
    }

    public String rate_string(ScStatEntry base) {
        return short_name + " / " + base.short_name + ": " + rate(base);
    }

    public void add(long v) {
        ctr.addAndGet(v);
    }
}
