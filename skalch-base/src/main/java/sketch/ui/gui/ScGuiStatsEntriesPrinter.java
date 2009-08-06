package sketch.ui.gui;

import sketch.dyn.stats.ScStatsPrinter;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.modifiers.ScUiModifierInner;

public class ScGuiStatsEntriesPrinter extends ScModifierDispatcher implements
        ScStatsPrinter
{
    public String stats_text;

    public ScGuiStatsEntriesPrinter(ScGuiStatsWarningsPrinter warnings,
            ScUiThread ui_thread)
    {
        super(ui_thread, null);
        stats_text = "<p><b>statistics</b></p><ul>";
        if (warnings.any_warnings) {
            stats_text = warnings.warnings + stats_text;
        }
    }

    @Override
    public String toString() {
        return "ScStatsEntriesPrinter [stats_text=" + stats_text + "]";
    }

    public void print_stat_line(String line) {
        stats_text += "<li>" + line + "</li>";
    }

    public void print_stat_warning(String line) {
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    private class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.statsEditor.setText("<html><body>" + stats_text
                    + "</ul></body></html>");
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }
}
