package sketch.dyn.ctrls;

import sketch.ui.sourcecode.ScSourceLocation;

/**
 * location of a rich construct in the source
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScCtrlSourceInfo implements Comparable<ScCtrlSourceInfo> {
    public ScCtrlConstructInfo info;
    public ScSourceLocation src_loc;

    public ScCtrlSourceInfo(ScCtrlConstructInfo info, ScSourceLocation src_loc)
    {
        this.info = info;
        this.src_loc = src_loc;
    }

    @Override
    public String toString() {
        return src_loc.toString();
    }

    public int compareTo(ScCtrlSourceInfo o) {
        return src_loc.compareTo(o.src_loc);
    }
}
