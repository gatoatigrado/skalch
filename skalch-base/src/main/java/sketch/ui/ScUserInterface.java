package sketch.ui;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
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

    public void addStackSolution(ScStack stack);

    public void set_counterexamples(ScSolvingInputConf[] inputs);

    public void addGaSynthesis(ScGaSynthesis sc_ga_synthesis);

    /** as with stack synthesis, the individual has not been cloned yet */
    public void addGaSolution(ScGaIndividual individual);

    public void displayAnimated(ScGaIndividual individual);
}
