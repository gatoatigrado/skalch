package sketch.ui.sourcecode;

import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;

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

    public String visitHoleInfo(ScSourceConstruct ctrlSrcInfo) {
        return ScSourceCache.singleton().getSourceString(
                ctrlSrcInfo.argumentLocation);
    }
}
