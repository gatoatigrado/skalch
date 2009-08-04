package sketch.ui.sourcecode;

import sketch.dyn.main.old.ScOldDynamicSketch;

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
    ScOldDynamicSketch sketch;

    public ScSourceUntilvHole(int uid, ScOldDynamicSketch sketch) {
        this.uid = uid;
        this.sketch = sketch;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    public String valueString(String src_args) {
        try {
            return sketch.ctrl_conf.getValueString(uid).formatString();
        } catch (ScNoValueStringException e) {
            return "/* not reached */ ??(" + src_args + ")";
        }
    }

    public String getName() {
        return "UntilvHole[uid=" + uid + "]";
    }
}
