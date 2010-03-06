package sketch.ui;

import static sketch.util.DebugOut.BASH_SALMON;
import static sketch.util.DebugOut.print_colored;
import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.stats.ScStatsPrinter;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

/**
 * the null user interface.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScDebugConsoleUI implements ScUserInterface, ScStatsPrinter {
    ScFixedInputConf[] all_counterexamples;
    ScDynamicSketchCall<?> ui_sketch;
    public final BackendOptions be_opts;

    public ScDebugConsoleUI(BackendOptions be_opts, ScDynamicSketchCall<?> sketch) {
        this.be_opts = be_opts;
        ui_sketch = sketch;
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
        if (be_opts.ui_opts.no_con_skdprint) {
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
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[skdprint]", " ", false,
                    debug_entry.consoleString());
        }
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (be_opts.ui_opts.print_counterex) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[user requested print]", "\n",
                    true, text);
        }
        all_counterexamples = ScFixedInputConf.from_inputs(inputs);
    }

    public void setStats(ScStatsModifier modifier) {
        print_stat_line("=== statistics ===");
        modifier.execute(this);
    }

    public void print_stat_line(String line) {
        print_colored(BASH_SALMON, "[stats]", "", false, line);
    }

    public void print_stat_warning(String line) {
        DebugOut.not_implemented("ScStatsPrinter.print_stat_warning");
    }

    public void synthesisFinished() {
        DebugOut.print("Finished synthesizing");

    }
}
