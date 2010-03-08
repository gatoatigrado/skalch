package sketch.dyn.constructs.ctrls;

import java.util.Vector;

import sketch.dyn.synth.stack.ScStack;
import sketch.ui.sourcecode.ScConstructValue;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScHighlightValues;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.wrapper.ScRichString;

/**
 * Wrapper for an array of holes.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScSynthCtrlConf extends ScCtrlConf {
    public ScStack stack;
    public int logType;
    public int[] values; // -1 for not accessed
    public int[] untilv;
    public int[] setCnt;
    public ScConstructValueString[] valueString;

    public ScSynthCtrlConf(ScStack stack, int logType) {
        this.stack = stack;
        this.logType = logType;
        values = new int[0];
        untilv = new int[0];
        setCnt = new int[0];
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (int value : values) {
            result *= 3333;
            result += value;
        }
        return result;
    }

    @Override
    public String toString() {
        String[] valuesStr = new String[values.length];
        for (int a = 0; a < values.length; a++) {
            valuesStr[a] = String.valueOf(values[a]);
        }
        return "ScSynthCtrlConf[" + (new ScRichString(", ")).join(valuesStr) + "]";
    }

    public void realloc(int minLength) {
        int nextLength = Math.max(minLength, values.length * 2);
        int[] nextValues = new int[nextLength];
        int[] nextUntilv = new int[nextLength];
        int[] nextSetCnt = new int[nextLength];
        System.arraycopy(values, 0, nextValues, 0, values.length);
        System.arraycopy(untilv, 0, nextUntilv, 0, untilv.length);
        System.arraycopy(setCnt, 0, nextSetCnt, 0, setCnt.length);
        for (int a = values.length; a < nextLength; a++) {
            nextValues[a] = -1;
            nextUntilv[a] = -1;
        }
        values = nextValues;
        untilv = nextUntilv;
        setCnt = nextSetCnt;
    }

    public boolean set(int uid, int v) {
        setCnt[uid] += 1;
        if (v < untilv[uid]) {
            values[uid] = v;
            return true;
        } else {
            return false;
        }
    }

    public void resetAccessed(int uid) {
        values[uid] = -1;
    }

    public void copyValuesFrom(ScSynthCtrlConf prev) {
        values = prev.values.clone();
        untilv = prev.untilv.clone();
        setCnt = prev.setCnt.clone();
    }

    @Override
    public int getValue(int uid) {
        if (values[uid] == -1) {
            values[uid] = 0;
            stack.addEntry(logType, uid, 0);
        }
        return values[uid];
    }

    public void generateValueStrings() {
        Vector<ScHighlightValues.Value> valueArr = new Vector<ScHighlightValues.Value>();
        for (int a = 0; a < values.length; a++) {
            ScConstructValue value = new ScConstructValue(Math.max(0, values[a]));
            valueArr.add(new ScHighlightValues.Value(value, setCnt[a], null));
        }
        valueString = new ScConstructValueString[values.length];
        ScHighlightValues.genValueStrings(valueArr);
        for (int a = 0; a < values.length; a++) {
            valueString[a] = valueArr.get(a).result;
        }
    }

    @Override
    public ScConstructValueString getValueString(int uid) throws ScNoValueStringException
    {
        if (uid >= valueString.length) {
            throw new ScNoValueStringException();
        }
        return valueString[uid];
    }

    @Override
    public int getDynamicValue(int uid, int untilv) {
        if (uid >= values.length) {
            realloc(uid + 1);
        }
        this.untilv[uid] = untilv;
        return getValue(uid);
    }

    @Override
    public int[] getValueArray() {
        return values;
    }
}
