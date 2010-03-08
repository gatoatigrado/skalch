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

    public void printEntry(int align, ScStatEntry entry) {
        if (showZero || entry.value > 0) {
            lines.add(entry.formatString(align));
        }
    }

    public void printEntries(ScStatEntry... entries) {
        int align = 0;
        for (ScStatEntry entry : entries) {
            printEntry(align, entry);
            align = 30;
        }
    }

    public void printRate(String indent, ScStatEntry entry, ScStatEntry base) {
        if (showZero || entry.value > 0) {
            lines.add(indent + entry.rateString(base));
        }
    }

    /** Main call by UI's */
    public void execute(ScStatsPrinter printer) {
        if (printer == null) {
            assertFalse("ScStatsModifier -- argument $printer$ null");
        }
        for (String warning : warnings) {
            printer.printStatWarning(warning);
        }
        for (String line : lines) {
            printer.printStatLine(line);
        }
    }

    public void rateWarnGt(ScStatEntry numer, ScStatEntry base,
            float trigger, String message)
    {
        if ((numer.getValue() != 0) && (numer.rate(base) > trigger)) {
            warnings.add(message);
        }
    }

    public void rateWarnLt(ScStatEntry numer, ScStatEntry base,
            float trigger, String message)
    {
        if ((numer.getValue() != 0) && (numer.rate(base) < trigger)) {
            warnings.add(message);
        }
    }
}
