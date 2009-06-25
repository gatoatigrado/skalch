package sketch.dyn.inputs;

public abstract class ScInputConf {
    public abstract int nextValue(int uid);

    public abstract int dynamicNextValue(int uid, int untilv);
}
