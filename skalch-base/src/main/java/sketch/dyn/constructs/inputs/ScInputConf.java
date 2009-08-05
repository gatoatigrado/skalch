package sketch.dyn.constructs.inputs;

import java.util.Vector;

import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;

public abstract class ScInputConf {
    public abstract int nextValue(int uid);

    public abstract int dynamicNextValue(int uid, int untilv);

    public abstract Vector<ScConstructValueString> getValueString(int uid)
            throws ScNoValueStringException;
}
