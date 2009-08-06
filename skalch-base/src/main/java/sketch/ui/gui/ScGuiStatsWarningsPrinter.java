package sketch.ui.gui;

import sketch.dyn.stats.ScStatsPrinter;

public class ScGuiStatsWarningsPrinter implements ScStatsPrinter {
    public String warnings;
    public boolean any_warnings = false;

    public ScGuiStatsWarningsPrinter() {
        warnings = "";
    }

    public void print_stat_line(String line) {
    }

    public void print_stat_warning(String line) {
        any_warnings = true;
        warnings += line + "<br />";
    }
}
