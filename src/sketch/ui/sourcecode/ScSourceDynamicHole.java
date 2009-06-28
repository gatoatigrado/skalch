package sketch.ui.sourcecode;

import sketch.dyn.ScDynamicSketch;

/**
 * An outer class which is responsible for formatting holes in the sketch.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceDynamicHole implements ScSourceConstructInfo {
    int uid;
    ScDynamicSketch sketch;

    public ScSourceDynamicHole(int uid, ScDynamicSketch sketch) {
        this.uid = uid;
        this.sketch = sketch;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    public String valueString() {
        return sketch.ctrl_conf.getValueString(uid);
    }

    public String formatSolution(String srcArgs) {
        return "formatSolution of ScSourceDynamicHole";
    }
}
