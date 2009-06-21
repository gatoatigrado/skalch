package sketch.ui;

/**
 * A class which has bound variables. This is useful for the synthesis list,
 * where different synthesizers may have significantly different functionalities
 * for showing in-progress synthesis.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScUiModifierDispatcher {
    public abstract String toString();

    public abstract ScUiModifier dispatch();
}
