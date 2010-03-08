package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.util.wrapper.ScRichString;

/**
 * true / false (parameterless call) to an oracle.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceBooleanOracle extends ScSourceUntilvOracle {
    public ScSourceBooleanOracle(int uid, ScDynamicSketchCall<?> sketchCall) {
        super(uid, sketchCall);
    }

    @Override
    public String getName() {
        return "BooleanOracle[uid=" + uid + "]";
    }

    @Override
    public String valueString(String srcArgs) {
        try {
            Vector<ScConstructValueString> values =
                    sketchCall.getOracleValueString(uid);
            Vector<String> resultArr = new Vector<String>();
            for (ScConstructValueString valueString : values) {
                boolean v = valueString.value.intValue() == 1;
                resultArr.add(valueString.formatWithNewValueString(v ? "true"
                        : "false"));
            }
            return (new ScRichString(", ")).join(resultArr
                    .toArray(new String[0]));
        } catch (ScNoValueStringException e) {
            return "not encountered";
        }
    }
}
