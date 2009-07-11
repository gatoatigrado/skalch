package sketch.ui;

import sketch.dyn.BackendOptions;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.debug.ScDebugEntry;
import sketch.dyn.debug.ScDebugSketchRun;
import sketch.dyn.ga.ScGaIndividual;
import sketch.dyn.ga.ScGaSynthesis;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stack.ScLocalStackSynthesis;
import sketch.dyn.stack.ScStack;
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
    ScFixedInputConf[] all_counterexamples;
    ScDynamicSketch ui_sketch;

    public ScDebugConsoleUI(ScDynamicSketch ui_sketch) {
        this.ui_sketch = ui_sketch;
    }

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

    public void addStackSolution(ScStack stack__, int solution_cost) {
        DebugOut.print_mt("solution with stack", stack__);
        if (BackendOptions.ui_opts.bool_("no_console_skdprint")) {
            return;
        }
        ScStack stack = stack__.clone();
        ScDebugSketchRun sketch_run =
                new ScDebugSketchRun(ui_sketch, stack, all_counterexamples);
        sketch_run.run();
        for (ScDebugEntry debug_entry : sketch_run.debug_out) {
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[skdprint]", " ",
                    false, debug_entry.consoleString());
        }
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (BackendOptions.ui_opts.bool_("print_counterexamples")) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN,
                    "[user requested print]", "\n", true, text);
        }
        all_counterexamples = ScFixedInputConf.from_inputs(inputs);
    }

    public void addGaSynthesis(ScGaSynthesis sc_ga_synthesis) {
        DebugOut.todo("add ga synthesis for debug");
    }

    public void addGaSolution(ScGaIndividual individual) {
        DebugOut.todo("solution ga synthesis individual", individual);
    }
}
