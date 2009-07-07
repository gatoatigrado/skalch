package sketch.dyn.debug;

import java.util.LinkedList;

/**
 * An entry used to identify either code hits or as a context for other debug
 * statements.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocationDebugEntry extends ScDebugEntry {
    public String location;

    public ScLocationDebugEntry(String location) {
        this.location = location;
    }

    @Override
    public String consoleString() {
        return "@ " + location + ": ";
    }

    @Override
    public String htmlString(LinkedList<String> active_html_contexts) {
        String rv = "<p><i>@ " + location + ":</i></p>";
        if (active_html_contexts.contains("ul")) {
            rv = "</ul>" + rv + "<ul>";
        }
        return rv;
    }
}
