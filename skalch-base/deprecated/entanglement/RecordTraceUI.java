package sketch.ui.entanglement;

import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;

public class RecordTraceUI implements ScUserInterface {

    public final ScUserInterface base;

    private Vector<Trace> traces;
    private boolean isFinished;

    public RecordTraceUI(ScUserInterface base) {
        this.base = base;
        isFinished = false;

        traces = new Vector<Trace>();
    }

    public void addStackSolution(ScStack stack) {
        if (traces != null && !isFinished) {
            traces.add(stack.getExecutionTrace());
        }
        base.addStackSolution(stack);
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSsr) {
        base.addStackSynthesis(localSsr);
    }

    public void modifierComplete(ScUiModifier m) {
        base.modifierComplete(m);
    }

    public int nextModifierTimestamp() {
        return base.nextModifierTimestamp();
    }

    public void setStats(ScStatsModifier modifier) {
        base.setStats(modifier);
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        base.set_counterexamples(inputs);
    }

    public void synthesisFinished() {
        EntanglementConsole con = new EntanglementConsole(traces);
        con.startConsole();
        base.synthesisFinished();
        isFinished = true;
    }
}
