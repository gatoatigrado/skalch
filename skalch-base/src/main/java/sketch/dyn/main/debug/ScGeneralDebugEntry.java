package sketch.dyn.main.debug;

import java.util.LinkedList;

import sketch.util.fcns.ScHtmlUtil;

public class ScGeneralDebugEntry extends ScDebugEntry {
    String text;

    public ScGeneralDebugEntry(String text) {
        this.text = text;
    }

    @Override
    public String consoleString() {
        return text;
    }

    @Override
    public String htmlString(LinkedList<String> activeHtmlContexts) {
        String rv = ScHtmlUtil.html_nonpre_code(text);
        rv = "<li>" + rv + "</li>";
        if (!activeHtmlContexts.contains("ul")) {
            rv = "<ul>" + rv + "</ul>";
        }
        return rv + "\n";
    }
}
