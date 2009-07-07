package sketch.dyn.debug;

import java.util.LinkedList;

/**
 * an entry logged by $skdprint$ and friends.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDebugEntry {
    public abstract String consoleString();

    /** TODO - make this some sort of visitor pattern */
    public abstract String htmlString(LinkedList<String> active_html_contexts);
}
