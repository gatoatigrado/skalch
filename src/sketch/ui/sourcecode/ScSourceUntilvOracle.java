package sketch.ui.sourcecode;

import sketch.dyn.ScDynamicSketch;

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

    public String valueString(String srcArgs) {
        return sketch.oracle_input_backend.getValueString(uid);
    }
}
