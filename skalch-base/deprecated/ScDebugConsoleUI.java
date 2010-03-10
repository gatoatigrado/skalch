package sketch.ui;

import static sketch.util.DebugOut.BASH_SALMON;
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
import sketch.result.ScSynthesisResults;
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
    ScFixedInputConf[] allCounterexamples;
    ScDynamicSketchCall<?> uiSketch;
    public final BackendOptions beOpts;

    public ScDebugConsoleUI(BackendOptions beOpts, ScDynamicSketchCall<?> sketch) {
        this.beOpts = beOpts;
        uiSketch = sketch;
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSsr) {
        DebugOut.print("ui add stack synthesis", localSsr);
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
        if (beOpts.uiOpts.noConSkdprint) {
            return;
        }
        ScStack stack = stack__.clone();
        // FIXME -- hack
        printDebugRun(new ScDebugStackRun(
                (ScDynamicSketchCall<ScAngelicSketchBase>) uiSketch, stack));
    }

    protected void printDebugRun(ScDebugRun sketchRun) {
        sketchRun.run();
        if (sketchRun.debugOut == null) {
            DebugOut.assertFalse("debug out is null");
        }
        for (ScDebugEntry debugEntry : sketchRun.debugOut) {
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[skdprint]", " ", false,
                    debugEntry.consoleString());
        }
    }

    public void setCounterexamples(ScSolvingInputConf[] inputs) {
        if (beOpts.uiOpts.printCounterex) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[user requested print]", "\n",
                    true, text);
        }
        allCounterexamples = ScFixedInputConf.fromInputs(inputs);
    }

    public void setStats(ScStatsModifier modifier) {
        printStatLine("=== statistics ===");
        modifier.execute(this);
    }

    public void printStatLine(String line) {
        DebugOut.print_colored(BASH_SALMON, "[stats]", "", false, line);
    }

    public void printStatWarning(String line) {
        DebugOut.not_implemented("ScStatsPrinter.printStatWarning");
    }

    public void synthesisFinished() {
        DebugOut.print("Finished synthesizing");

    }

    public void resetStackSolutions() {
    // TODO Auto-generated method stub

    }

    public void resetStackSyntheses() {
    // TODO Auto-generated method stub

    }

    public void setScSynthesisResults(ScSynthesisResults results) {
    // TODO Auto-generated method stub

    }

}
