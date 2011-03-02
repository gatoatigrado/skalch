package sketch.dyn.main.debug;

import java.awt.Color;
import java.util.LinkedList;

import sketch.ui.sourcecode.ScHighlightValues;
import sketch.util.fcns.ScHtmlUtil;

public class ScGeneralDebugEntry extends ScDebugEntry {
    private final String text;
    private final Color color;

    public ScGeneralDebugEntry(String text) {
        this.text = text;
        color = Color.black;
    }

    public ScGeneralDebugEntry(String text, Color color) {
        this.text = text;
        this.color = color;
    }

    @Override
    public String consoleString() {
        return text;
    }

    @Override
    public String htmlString(LinkedList<String> activeHtmlContexts) {
        String rv = ScHtmlUtil.html_nonpre_code(text);
        rv = "<li>" + colorString(rv) + "</li>";
        if (!activeHtmlContexts.contains("ul")) {
            rv = "<ul>" + rv + "</ul>";
        }
        return rv + "\n";
    }

    public String colorString(String original) {
        String stringColor = ScHighlightValues.getColorString(color);
        return "<span style=\"color:" + stringColor + "\">" + original + "</span>";
    }
}
