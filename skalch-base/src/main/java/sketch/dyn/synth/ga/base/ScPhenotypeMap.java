package sketch.dyn.synth.ga.base;

import java.util.Vector;

import sketch.util.DebugOut;
import sketch.util.ScCloneable;

/**
 * maps (type, uid, subuid) tuples to genotype entries, which are an array of
 * integers.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPhenotypeMap implements ScCloneable<ScPhenotypeMap> {
    public Vector<ScPhenotypeEntry> entries;
    public int spine_mask;
    public int spine_length;

    public ScPhenotypeMap(int spine_length) {
        if ((spine_length & (spine_length - 1)) > 0) {
            DebugOut.assertFalse("spine length must be a power of 2");
        }
        spine_mask = (spine_length - 1);
        this.spine_length = spine_length;
        entries = new Vector<ScPhenotypeEntry>(2 * spine_length);
    }

    @Override
    public String toString() {
        return "ScPhenotypeMap [entries=" + entries + ", spine_length="
                + spine_length + "]";
    }

    @Override
    @SuppressWarnings("unchecked")
    public ScPhenotypeMap clone() {
        ScPhenotypeMap next = new ScPhenotypeMap(spine_length);
        next.entries = (Vector<ScPhenotypeEntry>) entries.clone();
        return next;
    }

    public int get_index(boolean type, int uid, int subuid) {
        int idx = get_hash(type, uid, subuid);
        while (true) {
            if (idx >= entries.size()) {
                entries.setSize(2 * idx + 1);
            }
            ScPhenotypeEntry entry = entries.get(idx);
            if (entry == null) {
                entries.set(idx, new ScPhenotypeEntry(type, uid, subuid, idx
                        + spine_length));
                break;
            } else if (entry.match(type, uid, subuid)) {
                break;
            }
            idx = entry.next;
        }
        return idx;
    }

    public int get_hash(boolean type, int uid, int subuid) {
        return (uid * 36061 + subuid * 64063 + (type ? 1 : 0)) & spine_mask;
    }

    public String formatValuesString(ScGenotype genotype) {
        StringBuilder result = new StringBuilder();
        for (int a = 0; a < entries.size(); a++) {
            ScPhenotypeEntry entry = entries.get(a);
            if (entry != null) {
                result.append("    at " + a + ": " + entry.toString() + " = "
                        + genotype.formatIndex(a) + "\n");
            }
        }
        return result.toString();
    }
}
