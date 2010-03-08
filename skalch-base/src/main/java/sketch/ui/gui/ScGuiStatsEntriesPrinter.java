package sketch.ui.gui;

import sketch.dyn.stats.ScStatsPrinter;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.modifiers.ScUiModifierInner;

public class ScGuiStatsEntriesPrinter extends ScModifierDispatcher implements
        ScStatsPrinter
{
    public String statsText;

    public ScGuiStatsEntriesPrinter(ScGuiStatsWarningsPrinter warnings,
            ScUiThread uiThread)
    {
        super(uiThread, null);
        statsText = "<p><b>statistics</b></p><ul>";
        if (warnings.anyWarnings) {
            statsText = warnings.warnings + statsText;
        }
    }

    @Override
    public String toString() {
        return "ScStatsEntriesPrinter [statsText=" + statsText + "]";
    }

    public void printStatLine(String line) {
        statsText += "<li>" + line + "</li>";
    }

    public void printStatWarning(String line) {}

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    private class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            uiThread.gui.statsEditor.setText("<html><body>" + statsText +
                    "</ul></body></html>");
        }
    }

    @Override
    public ScUiModifierInner getModifier() {
        return new Modifier();
    }
}
