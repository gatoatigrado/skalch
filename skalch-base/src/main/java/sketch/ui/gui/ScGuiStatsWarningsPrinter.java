package sketch.ui.gui;

import sketch.dyn.stats.ScStatsPrinter;

public class ScGuiStatsWarningsPrinter implements ScStatsPrinter {
    public String warnings;
    public boolean anyWarnings = false;

    public ScGuiStatsWarningsPrinter() {
        warnings = "";
    }

    public void printStatLine(String line) {
    }

    public void printStatWarning(String line) {
        anyWarnings = true;
        warnings += line + "<br />";
    }
}
