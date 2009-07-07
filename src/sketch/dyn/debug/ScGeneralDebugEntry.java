package sketch.dyn.debug;

import java.util.LinkedList;

import sketch.ui.sourcecode.ScHighlightSourceVisitor;

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
    public String htmlString(LinkedList<String> active_html_contexts) {
        String rv = ScHighlightSourceVisitor.html_nonpre_code(text);
        rv = "<li>" + rv + "</li>";
        if (!active_html_contexts.contains("ul")) {
            rv = "<ul>" + rv + "</ul>";
        }
        return rv + "\n";
    }
}
