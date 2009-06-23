package sketch.ui;

import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.ui.modifiers.ScUiModifier;

/**
 * shared functions for ui
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public interface ScUserInterface {
    public void modifierComplete(ScUiModifier m);

    public int nextModifierTimestamp();

    public void addStackSynthesis(ScLocalStackSynthesis local_ssr);

    public void addSolution(ScStack stack);
}
