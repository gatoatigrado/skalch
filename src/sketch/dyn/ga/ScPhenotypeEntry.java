package sketch.dyn.ga;

/**
 * meta-info about a phenotype entry; it's index is the same as the
 * corresponding genotype entry.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPhenotypeEntry {
    public boolean type;
    public int uid;
    public int subuid;
    public int next;

    public ScPhenotypeEntry(boolean type, int uid, int subuid, int next) {
        this.type = type;
        this.uid = uid;
        this.subuid = subuid;
        this.next = next;
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
}
