package sketch.dyn.constructs.ctrls;

import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;

public abstract class ScCtrlConf {
    public abstract int[] getValueArray();

    public abstract int getValue(int uid);

    public abstract int getDynamicValue(int uid, int untilv);

    public abstract ScConstructValueString getValueString(int uid)
            throws ScNoValueStringException;
}
