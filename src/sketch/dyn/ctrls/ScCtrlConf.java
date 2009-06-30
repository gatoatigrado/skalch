package sketch.dyn.ctrls;

import sketch.ui.sourcecode.ScNoValueStringException;

public abstract class ScCtrlConf {
    public abstract int getValue(int uid);

    public abstract int getDynamicValue(int uid, int untilv);

    public abstract String getValueString(int uid)
            throws ScNoValueStringException;
}
