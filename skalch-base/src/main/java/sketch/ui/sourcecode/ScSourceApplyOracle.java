package sketch.ui.sourcecode;

import sketch.dyn.main.ScDynamicSketchCall;

/**
 * Oracle which (if it had been a hole) would be rewritten as (arg)(!!-value)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceApplyOracle extends ScSourceUntilvOracle {
    public ScSourceApplyOracle(int uid, ScDynamicSketchCall<?> sketchCall) {
        super(uid, sketchCall);
    }

    @Override
    public String getName() {
        return "ApplyOracle[uid=" + uid + "]";
    }
}
