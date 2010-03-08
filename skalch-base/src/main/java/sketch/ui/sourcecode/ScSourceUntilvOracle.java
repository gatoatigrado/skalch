package sketch.ui.sourcecode;

import java.util.Vector;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.util.wrapper.ScRichString;

/**
 * an oracle which (if it were a hole) would be rewritten to a single number.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceUntilvOracle implements ScSourceConstructInfo {
    int uid;
    ScDynamicSketchCall<?> sketchCall;

    public ScSourceUntilvOracle(int uid, ScDynamicSketchCall<?> sketchCall) {
        this.uid = uid;
        this.sketchCall = sketchCall;
    }

    public boolean hasMultipleValues() {
        return false;
    }

    /** the source location visitor will add the comment above */
    public String valueString(String srcArgs) {
        try {
            Vector<ScConstructValueString> values =
                    sketchCall.getOracleValueString(uid);
            Vector<String> resultArr = new Vector<String>();
            for (ScConstructValueString valueString : values) {
                resultArr.add(valueString.formatString());
            }
            return (new ScRichString(", ")).join(resultArr
                    .toArray(new String[0]));
        } catch (ScNoValueStringException e) {
            return "not encountered";
        }
    }

    public String getName() {
        return "UntilvOracle[uid=" + uid + "]";
    }
}
