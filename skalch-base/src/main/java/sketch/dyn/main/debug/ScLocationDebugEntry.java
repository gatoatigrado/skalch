package sketch.dyn.main.debug;

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
    public String htmlString() {
        return "<p><i>@ " + location + ":</i></p>";
    }

    @Override
    public boolean hasEndline() {
        return true;
    }
}
