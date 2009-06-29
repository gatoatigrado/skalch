package sketch.ui.sourcecode;

import sketch.dyn.ScDynamicSketch;

/**
 * an oracle which (if it were a hole) would be rewritten to a single number.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceUntilvOracle implements ScSourceConstructInfo {
    int uid;
    ScDynamicSketch sketch;

    public ScSourceUntilvOracle(int uid, ScDynamicSketch sketch) {
        this.uid = uid;
        this.sketch = sketch;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    /** the source location visitor will add the comment above */
    public String valueString(String srcArgs) {
        return sketch.oracle_conf.getValueString(uid);
    }

    public String getName() {
        return "UntilvOracle[uid=" + uid + "]";
    }
}
