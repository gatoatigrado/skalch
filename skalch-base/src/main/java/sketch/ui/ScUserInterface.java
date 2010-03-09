package sketch.ui;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.modifiers.ScUiModifier;

/**
 * shared functions for ui
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public interface ScUserInterface {
    public void modifierComplete(ScUiModifier m);

    public int nextModifierTimestamp();

    public void addStackSynthesis(ScLocalStackSynthesis localSsr);

    public void resetStackSyntheses();

    public void addStackSolution(ScStack stack);

    public void resetStackSolutions();

    public void setCounterexamples(ScSolvingInputConf[] inputs);

    public void setStats(ScStatsModifier modifier);

    public void synthesisFinished();
}
