package sketch.ui.sourcecode;

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

    public String visitHoleInfo(ScSourceConstruct ctrl_src_info) {
        return ScSourceCache.singleton().getSourceString(
                ctrl_src_info.argument_location);
    }
}
