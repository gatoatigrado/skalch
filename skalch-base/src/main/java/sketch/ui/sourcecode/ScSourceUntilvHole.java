package sketch.ui.sourcecode;

import sketch.dyn.main.ScDynamicSketchCall;

/**
 * An class which is responsible for formatting untilv holes (holes with an
 * integer untilv parameter) in the sketch.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceUntilvHole implements ScSourceConstructInfo {
    int uid;
    ScDynamicSketchCall<?> sketchCall;

    public ScSourceUntilvHole(int uid, ScDynamicSketchCall<?> sketchCall) {
        this.uid = uid;
        this.sketchCall = sketchCall;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    public String valueString(String srcArgs) {
        try {
            return sketchCall.getHoleValueString(uid).formatString();
        } catch (ScNoValueStringException e) {
            return "/* not reached */ ??(" + srcArgs + ")";
        }
    }

    public String getName() {
        return "UntilvHole[uid=" + uid + "]";
    }
}
