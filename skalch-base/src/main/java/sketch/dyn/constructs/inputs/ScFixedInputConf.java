package sketch.dyn.constructs.inputs;

import java.util.Vector;

import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScHighlightValues;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.DebugOut;
import sketch.util.wrapper.ScRichString;

/**
 * Static inputs that may have slightly faster access. This is also used for formatting
 * values according to their access frequency.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScFixedInputConf extends ScInputConf {
    protected int[][] values;
    protected int[][] setCnt;
    protected int[] untilv;
    protected int[] next;
    public Vector<ScConstructValueString>[] valueString;

    public ScFixedInputConf(int[][] values, int[][] setCnt, int[] untilv, int[] next) {
        this.values = values;
        this.setCnt = setCnt;
        this.untilv = untilv;
        this.next = next;
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (int[] vArr : values) {
            result *= 13;
            for (int v : vArr) {
                result *= 344;
                result += v;
            }
        }
        return result;
    }

    @Override
    public String toString() {
        String[] valuesStr = new String[values.length];
        for (int a = 0; a < values.length; a++) {
            valuesStr[a] = "";
            for (int c = 0; c < values[a].length; c++) {
                valuesStr[a] += (c == 0 ? "" : ", ") + values[a][c];
            }
        }
        return "ScFixedInputConf[ " + (new ScRichString(", ")).join(valuesStr) + " ]";
    }

    @Override
    public int dynamicNextValue(int uid, int untilv_) {
        if (untilv[uid] != untilv_) {
            DebugOut.assertFalse("set untilv for fixed input generators");
        }
        return nextValue(uid);
    }

    @Override
    public int nextValue(int uid) {
        final int uidNext = next[uid];
        if (uidNext >= values[uid].length) {
            DebugOut.assertFalse("fixed input generator exceeding length", uid, uidNext,
                    values[uid].length);
            return 0;
        } else {
            int rv = values[uid][uidNext];
            next[uid] += 1;
            return rv;
        }
    }

    public static ScFixedInputConf[] fromInputs(ScSolvingInputConf[] inputs) {
        ScFixedInputConf[] result = new ScFixedInputConf[inputs.length];
        for (int a = 0; a < inputs.length; a++) {
            result[a] = inputs[a].fixedInputs();
        }
        return result;
    }

    public void resetIndex() {
        for (int a = 0; a < next.length; a++) {
            next[a] = 0;
        }
    }

    public void setCnt(int uidIdx, int subuidIdx, int value) {
        if (setCnt.length > uidIdx && setCnt[uidIdx].length > subuidIdx) {
            setCnt[uidIdx][subuidIdx] = value;
        }
    }

    /** NOTE - a bit of messy (uid, subuid) -> index mapping */
    @SuppressWarnings("unchecked")
    public void generateValueStrings() {
        Vector<ScHighlightValues.Value> valueArr = new Vector<ScHighlightValues.Value>();
        for (int uidIdx = 0; uidIdx < values.length; uidIdx++) {
            for (int subuidIdx = 0; subuidIdx < values[uidIdx].length; subuidIdx++) {
                ScConstructValue value = new ScConstructValue(values[uidIdx][subuidIdx]);
                int colorV = setCnt[uidIdx][subuidIdx];
                int[] idArr = { uidIdx, subuidIdx };
                valueArr.add(new ScHighlightValues.Value(value, colorV, idArr));
            }
        }
        ScHighlightValues.genValueStrings(valueArr);
        // convert to a 2-d vector
        valueString = new Vector[values.length];
        for (int a = 0; a < valueString.length; a++) {
            valueString[a] = new Vector<ScConstructValueString>();
        }
        for (ScHighlightValues.Value value : valueArr) {
            int[] idArr = (int[]) value.tag;
            Vector<ScConstructValueString> uidV = valueString[idArr[0]];
            if (idArr[1] >= uidV.size()) {
                uidV.setSize(idArr[1] + 1);
            }
            uidV.set(idArr[1], value.result);
        }
    }

    @Override
    public Vector<ScConstructValueString> getValueString(int uid)
            throws ScNoValueStringException
    {
        if (uid >= valueString.length || valueString[uid].size() == 0) {
            throw new ScNoValueStringException();
        }
        return valueString[uid];
    }

    @Override
    public int[] getValueArray() {
        int nvalues = 0;
        for (int a = 0; a < values.length; a++) {
            nvalues += values[a].length;
        }
        int[] result = new int[nvalues];
        int idx = 0;
        for (int a = 0; a < values.length; a++) {
            for (int c = 0; c < values[a].length; c++) {
                result[idx] = values[a][c];
                idx += 1;
            }
        }
        return result;
    }
}
