package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.dyn.main.old.ScOldDynamicSketch;
import sketch.util.ScRichString;

/**
 * true / false (parameterless call) to an oracle.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceBooleanOracle extends ScSourceUntilvOracle {
    public ScSourceBooleanOracle(int uid, ScOldDynamicSketch sketch) {
        super(uid, sketch);
    }

    @Override
    public String getName() {
        return "BooleanOracle[uid=" + uid + "]";
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
            return (new ScRichString(", ")).join(result_arr
                    .toArray(new String[0]));
        } catch (ScNoValueStringException e) {
            return "not encountered";
        }
    }
}
