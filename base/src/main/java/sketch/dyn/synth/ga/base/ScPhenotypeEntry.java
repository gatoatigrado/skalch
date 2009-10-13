package sketch.dyn.synth.ga.base;

/**
 * meta-info about a phenotype entry; it's index is the same as the
 * corresponding genotype entry.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPhenotypeEntry implements Comparable<ScPhenotypeEntry> {
    /** true if control, false if oracle */
    public final boolean type;
    public final int uid;
    public final int subuid;
    public final int next;
    public final int index;

    public ScPhenotypeEntry(boolean type, int uid, int subuid, int next,
            int index)
    {
        this.type = type;
        this.uid = uid;
        this.subuid = subuid;
        this.next = next;
        this.index = index;
    }

    @Override
    public String toString() {
        return "ScPhenotypeEntry [type=" + type + ", uid=" + uid + ", subuid="
                + subuid + "]";
    }

    public boolean match(boolean type, int uid, int subuid) {
        return (this.type == type) && (this.uid == uid)
                && (this.subuid == subuid);
    }

    public boolean lessThan(ScPhenotypeEntry other) {
        if (other.uid < uid) {
            return true;
        } else if (other.uid == uid) {
            if (other.subuid < subuid) {
                return true;
            } else if (other.subuid == subuid && !other.type && type) {
                return true;
            }
        }
        return false;
    }

    public int compareTo(ScPhenotypeEntry other) {
        if (other.lessThan(this)) {
            return -1;
        } else if (lessThan(other)) {
            return 1;
        } else {
            return 0;
        }
    }
}
