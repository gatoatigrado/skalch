package sketch.ui.sourcecode;

import sketch.dyn.main.ScDynamicSketchCall;

/**
 * A class responsible for formatting holes parameterized by a collection.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceApplyHole extends ScSourceUntilvHole {
    public ScSourceApplyHole(int uid, ScDynamicSketchCall<?> sketchCall) {
        super(uid, sketchCall);
    }

    @Override
    public String valueString(String srcArgs) {
        try {
            return "(" + srcArgs + ")("
                    + sketchCall.getHoleValueString(uid).formatString() + ")";
        } catch (ScNoValueStringException e) {
            return "/* not reached */ ??(" + srcArgs + ")";
        }
    }

    @Override
    public String getName() {
        return "ApplyHole[uid=" + uid + "]";
    }
}
