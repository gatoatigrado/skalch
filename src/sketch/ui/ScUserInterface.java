package sketch.ui;

import sketch.dyn.ga.ScGaIndividual;
import sketch.dyn.ga.ScGaSynthesis;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stack.ScLocalStackSynthesis;
import sketch.dyn.stack.ScStack;
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

    public void addStackSolution(ScStack stack, int solution_cost);

    public void set_counterexamples(ScSolvingInputConf[] inputs);

    public void addGaSynthesis(ScGaSynthesis sc_ga_synthesis);

    public void addGaSolution(ScGaIndividual individual);
}
