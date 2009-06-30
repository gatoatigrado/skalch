package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.dyn.ScDynamicSketch;
import sketch.util.RichString;

public class ScSourceBooleanOracle extends ScSourceUntilvOracle {
    public ScSourceBooleanOracle(int uid, ScDynamicSketch sketch) {
        super(uid, sketch);
    }

    @Override
    public String valueString(String srcArgs) {
        try {
            Vector<ScConstructValueString> values =
                    sketch.oracle_conf.getValueString(uid);
            Vector<String> result_arr = new Vector<String>();
            for (ScConstructValueString value_string : values) {
                boolean v = value_string.value.intValue() == 1;
                result_arr.add(value_string.formatWithNewValueString(v ? "true"
                        : "false"));
            }
            return (new RichString(", ")).join(result_arr
                    .toArray(new String[0]));
        } catch (ScNoValueStringException e) {
            return "not encountered";
        }
    }
}
