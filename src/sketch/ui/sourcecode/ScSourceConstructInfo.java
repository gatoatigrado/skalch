package sketch.ui.sourcecode;

/**
 * An object representing a source construct object, that will format it's
 * completion. This is linked to a particular source location with
 * ScSourceConstruct. This could be a hole array object, regular holes, oracles,
 * and any other glorifications. These objects do not have to have a single or
 * static uid or untilv. Furthermore, a ScSourceConstructInfo could be an outer
 * class using a field to link it to a dynamic sketch. This is used for holes
 * replaced by the plugin.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract interface ScSourceConstructInfo {
    /**
     * can the construct return multiple values? if so, use multiline when
     * printing the string below
     */
    public boolean hasMultipleValues();

    public String valueString();

    public String formatSolution(String src_args);
}
