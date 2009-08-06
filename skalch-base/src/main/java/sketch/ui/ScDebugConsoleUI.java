package sketch.ui;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.assertFalse;
import sketch.dyn.constructs.ctrls.ScGaCtrlConf;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScGaInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScDebugGaRun;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
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
    ScDynamicSketchCall<?> ui_sketch;
    protected ScGaCtrlConf ga_ctrl_conf;
    protected ScGaInputConf ga_oracle_conf;

    public ScDebugConsoleUI(ScDynamicSketchCall<?> sketch) {
        ui_sketch = sketch;
        if (beopts().synth_opts.solver.isGa) {
            ga_ctrl_conf = new ScGaCtrlConf();
            ga_oracle_conf = new ScGaInputConf();
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

    @SuppressWarnings("unchecked")
    public void addStackSolution(ScStack stack__) {
        DebugOut.print_mt("solution with stack", stack__);
        if (beopts().ui_opts.no_console_skdprint) {
            return;
        }
        ScStack stack = stack__.clone();
        // FIXME -- hack
        printDebugRun(new ScDebugStackRun(
                (ScDynamicSketchCall<ScAngelicSketchBase>) ui_sketch, stack));
    }

    protected void printDebugRun(ScDebugRun sketch_run) {
        sketch_run.run();
        if (sketch_run.debug_out == null) {
            DebugOut.assertFalse("debug out is null");
        }
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

    @SuppressWarnings("unchecked")
    public void addGaSolution(ScGaIndividual individual) {
        DebugOut.print_mt("solution ga synthesis individual", individual);
        DebugOut.print_mt("solution population", individual.initial_population);
        ScGaIndividual clone = individual.clone();
        printDebugRun(new ScDebugGaRun(
                (ScDynamicSketchCall<ScAngelicSketchBase>) ui_sketch, clone,
                ga_ctrl_conf, ga_oracle_conf));
    }
}
