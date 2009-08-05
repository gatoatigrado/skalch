package sketch.ui.sourcecode;

/**
 * A formatted value, used to color holes and oracles according to how much they
 * were accessed.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScConstructValueString {
    public String formatTagsStart;
    public ScConstructValue value;
    public String formatTagsEnd;

    public ScConstructValueString(String formatTagsStart,
            ScConstructValue value, String formatTagsEnd)
    {
        this.formatTagsStart = formatTagsStart;
        this.value = value;
        this.formatTagsEnd = formatTagsEnd;
    }

    public String formatString() {
        return formatTagsStart + value.formatString() + formatTagsEnd;
    }

    public String formatWithNewValueString(String text) {
        return formatTagsStart + text + formatTagsEnd;
    }
}
