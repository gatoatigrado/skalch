package sketch.dyn.prefix;

import sketch.util.DebugOut;

/**
 * Not-yet-complete class which will represent a LocalPrefix converted to a
 * shared prefix.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSubtreePrefix extends ScSharedPrefix {
    public ScSubtreePrefix() {
        super(0);
        DebugOut.not_implemented("subtree prefix");
    }
}
