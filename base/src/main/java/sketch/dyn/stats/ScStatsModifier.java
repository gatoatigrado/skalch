package sketch.dyn.stats;

import static sketch.util.DebugOut.assertFalse;

import java.util.Vector;

/**
 * N.B. - not actually one of the GUI's modifiers, just somethign to hold calls
 * to a class implementing ScStatPrinter
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatsModifier {
    public Vector<String> lines = new Vector<String>();
    public Vector<String> warnings = new Vector<String>();
    protected final boolean showZero;

    public ScStatsModifier(boolean showZero) {
        this.showZero = showZero;
    }

    public void print_entry(int align, ScStatEntry entry) {
        if (showZero || entry.value > 0) {
            lines.add(entry.formatString(align));
        }
    }

    public void print_entries(ScStatEntry... entries) {
        int align = 0;
        for (ScStatEntry entry : entries) {
            print_entry(align, entry);
            align = 30;
        }
    }

    public void print_rate(String indent, ScStatEntry entry, ScStatEntry base) {
        if (showZero || entry.value > 0) {
            lines.add(indent + entry.rate_string(base));
        }
    }

    /** Main call by UI's */
    public void execute(ScStatsPrinter printer) {
        if (printer == null) {
            assertFalse("ScStatsModifier -- argument $printer$ null");
        }
        for (String warning : warnings) {
            printer.print_stat_warning(warning);
        }
        for (String line : lines) {
            printer.print_stat_line(line);
        }
    }

    public void rate_warn_gt(ScStatEntry numer, ScStatEntry base,
            float trigger, String message)
    {
        if ((numer.get_value() != 0) && (numer.rate(base) > trigger)) {
            warnings.add(message);
        }
    }

    public void rate_warn_lt(ScStatEntry numer, ScStatEntry base,
            float trigger, String message)
    {
        if ((numer.get_value() != 0) && (numer.rate(base) < trigger)) {
            warnings.add(message);
        }
    }
}
