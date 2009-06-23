package sketch.ui.sourcecode;

import sketch.dyn.ctrls.ScCtrlSourceInfo;

/**
 * visitor which will e.g. print a string corresponding to a source location
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScSourceLocationVisitor {
    public String visitCode(ScSourceLocation location) {
        return ScSourceCache.singleton().getSourceString(location);
    }

    public String visitHoleInfo(ScCtrlSourceInfo ctrl_src_info) {
        return ScSourceCache.singleton().getSourceString(ctrl_src_info.src_loc);
    }
}
