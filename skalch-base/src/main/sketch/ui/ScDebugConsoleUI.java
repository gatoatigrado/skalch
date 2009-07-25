package sketch.ui;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.assertFalse;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.debug.ScDebugEntry;
import sketch.dyn.debug.ScDebugRun;
import sketch.dyn.debug.ScDebugStackRun;
import sketch.dyn.ga.ScGaSynthesis;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScGaInputConf;
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
    protected ScGaCtrlConf ga_ctrl_conf;
    protected ScGaInputConf ga_oracle_conf;

    public ScDebugConsoleUI(ScDynamicSketch ui_sketch) {
        this.ui_sketch = ui_sketch;
        if (beopts().ga_opts.enable) {
            ga_ctrl_conf = new ScGaCtrlConf(ui_sketch.get_hole_info());
            ga_oracle_conf = new ScGaInputConf(ui_sketch.get_oracle_info());
        }
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
        if (beopts().ui_opts.no_console_skdprint) {
            return;
        }
        ScStack stack = stack__.clone();
        printDebugRun(new ScDebugStackRun(ui_sketch, stack, all_counterexamples));
    }

    protected void printDebugRun(ScDebugRun sketch_run) {
        sketch_run.run();
        for (ScDebugEntry debug_entry : sketch_run.debug_out) {
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[skdprint]", " ",
                    false, debug_entry.consoleString());
        }
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (beopts().ui_opts.print_counterexamples) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN,
                    "[user requested print]", "\n", true, text);
        }
        all_counterexamples = ScFixedInputConf.from_inputs(inputs);
    }

    public void displayAnimated(ScGaIndividual unused) {
        assertFalse("please use the GUI or disable ui_display_animated");
    }

    public void addGaSynthesis(ScGaSynthesis sc_ga_synthesis) {
        DebugOut.todo("add ga synthesis for debug");
    }

    public void addGaSolution(ScGaIndividual individual) {
        DebugOut.print_mt("solution ga synthesis individual", individual);
        DebugOut.print_mt("solution population", individual.initial_population);
        ScGaIndividual clone = individual.clone();
        printDebugRun(new sketch.dyn.debug.ScDebugGaRun(ui_sketch,
                all_counterexamples, clone, ga_ctrl_conf, ga_oracle_conf));
    }
}