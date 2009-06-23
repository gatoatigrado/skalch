package sketch.dyn.synth;

import sketch.util.DebugOut;
import sketch.util.ObjectFactory;
import sketch.util.ScCloneable;

/**
 * An entry of ScStack, contains fields to index holes or inputs.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public final class ScStackEntry implements ScCloneable<ScStackEntry> {
    public int type = -1;
    public int uid;
    public int subuid;

    public ScStackEntry(int type, int uid, int subuid) {
        set(type, uid, subuid);
    }

    public ScStackEntry() {
    }

    @Override
    public String toString() {
        if (type == -1) {
            return "<unset type>";
        } else if (type == ScStack.SYNTH_HOLE_LOG_TYPE) {
            return "hole " + uid;
        } else {
            if (type != ScStack.SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("ScStackEntry - unknown type");
            }
            return "oracle " + uid + "[" + subuid + "]";
        }
    }

    @Override
    public int hashCode() {
        return type + 13 * (uid + 13 * subuid);
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

    @Override
    public ScStackEntry clone() {
        return new ScStackEntry(type, uid, subuid);
    }

    public static class Factory extends ObjectFactory<ScStackEntry> {
        @Override
        public ScStackEntry create() {
            return new ScStackEntry();
        }
    }
}
