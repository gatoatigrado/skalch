package sketch.ui;

import sketch.dyn.BackendOptions;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

/**
 * the null user interface.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScDebugConsoleUI implements ScUserInterface {
    public void addStackSynthesis(ScLocalStackSynthesis local_ssr) {
        DebugOut.print("ui add stack synthesis", local_ssr);
    }

    public void modifierComplete(ScUiModifier m) {
        DebugOut.print("ui modifierComplete", m);
    }

    public int nextModifierTimestamp() {
        DebugOut.print("ui modifier timestamp");
        return 0;
    }

    public void addSolution(ScStack stack) {
        DebugOut.print_mt("solution with stack", stack);
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (BackendOptions.ui_opts.bool_("print_counterexamples")) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN,
                    "[user requested print]", "\n", true, text);
        }
    }
}
