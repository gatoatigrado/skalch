package sketch.dyn.main.debug;

import java.awt.Color;
import java.util.LinkedList;

import sketch.ui.sourcecode.ScHighlightValues;
import sketch.util.fcns.ScHtmlUtil;

public class ScGeneralDebugEntry extends ScDebugEntry {
    private final String text;
    private final Color color;
    private final boolean hasEndline;

    public ScGeneralDebugEntry(String text, boolean hasEndline, Color color) {
        this.text = text;
        this.hasEndline = hasEndline;
        this.color = color;
    }

    @Override
    public String consoleString() {
        return text;
    }

    @Override
    public String htmlString() {
        String rv = ScHtmlUtil.html_nonpre_code(text);
        return colorString(rv);
    }

    public String colorString(String original) {
        if (color == null) {
            return original;
        }
        String stringColor = ScHighlightValues.getColorString(color);
        return "<span style=\"color:" + stringColor + "\">" + original + "</span>";
    }

    @Override
    public boolean hasEndline() {
        return hasEndline;
    }
    
}
