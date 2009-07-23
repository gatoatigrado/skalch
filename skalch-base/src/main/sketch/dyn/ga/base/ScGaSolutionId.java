package sketch.dyn.ga.base;

import static sketch.util.DebugOut.assertFalse;

import java.util.Arrays;
import java.util.Vector;

/**
 * ensure that copies of the same solution aren't displayed.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaSolutionId {
    public Vector<ScGaSolutionIdEntry> entries =
            new Vector<ScGaSolutionIdEntry>();
    public ScGaSolutionIdEntry[] array;

    @Override
    public String toString() {
        return "ScGaSolutionId [array=" + Arrays.toString(array)
                + ", hashcode=" + hashCode() + "]";
    }

    @Override
    public int hashCode() {
        if (entries != null) {
            assertFalse("please call create_array() first");
        }
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(array);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (entries != null) {
            assertFalse("please call create_array() first");
        }
        if (obj == null || !(obj instanceof ScGaSolutionId)) {
            return false;
        }
        return Arrays.equals(array, ((ScGaSolutionId) obj).array);
    }

    public void create_array() {
        array = entries.toArray(new ScGaSolutionIdEntry[0]);
        Arrays.sort(array);
        entries = null;
    }

    public static class ScGaSolutionIdEntry implements
            Comparable<ScGaSolutionIdEntry>
    {
        Boolean type;
        Integer uid;
        Integer subuid;
        Integer value;

        public ScGaSolutionIdEntry(boolean type, int uid, int subuid, int value)
        {
            this.type = type;
            this.uid = uid;
            this.subuid = subuid;
            this.value = value;
        }

        @Override
        public String toString() {
            return "ScGaSolutionIdEntry [type=" + type + ", uid=" + uid
                    + ", subuid=" + subuid + ", value=" + value + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 16661;
            int result = 1;
            result =
                    prime * result + ((subuid == null) ? 0 : subuid.hashCode());
            result = prime * result + ((type == null) ? 0 : type.hashCode());
            result = prime * result + ((uid == null) ? 0 : uid.hashCode());
            result = prime * result + ((value == null) ? 0 : value.hashCode());
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof ScGaSolutionIdEntry)) {
                return false;
            }
            ScGaSolutionIdEntry other = (ScGaSolutionIdEntry) obj;
            return (type == other.type) && (uid == other.uid)
                    && (subuid == other.subuid) && (value == other.value);
        }

        public int compareTo(ScGaSolutionIdEntry o) {
            int typecmp = type.compareTo(o.type);
            if (typecmp != 0) {
                return typecmp;
            } else {
                int uidcmp = uid.compareTo(o.uid);
                if (uidcmp != 0) {
                    return uidcmp;
                }
            }
            return subuid.compareTo(o.subuid);
        }
    }
}
