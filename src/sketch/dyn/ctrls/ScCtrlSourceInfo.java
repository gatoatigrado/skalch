package sketch.dyn.ctrls;

import sketch.dyn.ScSourceLocation;

/**
 * location of a rich construct in the source
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScCtrlSourceInfo {
    public ScCtrlConstructInfo info;
    public ScSourceLocation src_loc;

    public ScCtrlSourceInfo(ScCtrlConstructInfo info, ScSourceLocation src_loc)
    {
        this.info = info;
        this.src_loc = src_loc;
    }
}
