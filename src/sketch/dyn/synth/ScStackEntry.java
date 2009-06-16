package sketch.dyn.synth;

/**
 * An entry of ScStack, contains fields to index holes or inputs.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public final class ScStackEntry {
    public int type = -1;
    public int uid;
    public int subuid;

    public ScStackEntry(int type, int uid, int subuid) {
        set(type, uid, subuid);
    }

    public ScStackEntry() {
    }

    public ScStackEntry copy() {
        ScStackEntry result = new ScStackEntry();
        result.set(type, uid, subuid);
        return result;
    }

    public void set(int type, int uid, int subuid) {
        this.type = type;
        this.uid = uid;
        this.subuid = subuid;
    }
}
