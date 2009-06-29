package sketch.ui.sourcecode;

import sketch.dyn.ScDynamicSketch;

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
    ScDynamicSketch sketch;

    public ScSourceUntilvHole(int uid, ScDynamicSketch sketch) {
        this.uid = uid;
        this.sketch = sketch;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    public String valueString(String src_args) {
        try {
            return sketch.ctrl_conf.getValueString(uid);
        } catch (ArrayIndexOutOfBoundsException e) {
            return "/* not reached */ ??(" + src_args + ")";
        }
    }

    public String getName() {
        return "UntilvHole[uid=" + uid + "]";
    }
}
