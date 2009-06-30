package sketch.dyn.inputs;

import sketch.ui.sourcecode.ScNoValueStringException;

public abstract class ScInputConf {
    public abstract int nextValue(int uid);

    public abstract int dynamicNextValue(int uid, int untilv);

    public abstract String getValueString(int uid)
            throws ScNoValueStringException;
}
