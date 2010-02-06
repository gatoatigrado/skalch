package sketch.ui.entanglement;

import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;

public class RecordTraceUI implements ScUserInterface {

    public final ScUserInterface base;

    private Vector<ExecutionTrace> listOfTraces;
    private boolean isFinished;

    public RecordTraceUI(ScUserInterface base) {
        this.base = base;
        isFinished = false;

        listOfTraces = new Vector<ExecutionTrace>();
    }

    public void addGaSolution(ScGaIndividual individual) {
        base.addGaSolution(individual);
    }

    public void addGaSynthesis(ScGaSynthesis scGaSynthesis) {
        base.addGaSynthesis(scGaSynthesis);
    }

    public void addStackSolution(ScStack stack) {
        if (listOfTraces != null && !isFinished) {
            listOfTraces.add(stack.getExecutionTrace());
        }
        base.addStackSolution(stack);
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSsr) {
        base.addStackSynthesis(localSsr);
    }

    public void displayAnimated(ScGaIndividual individual) {
        base.displayAnimated(individual);
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
        EntanglementAnalysis ea = new EntanglementAnalysis(listOfTraces);
        ea.findConstantElements();
        base.synthesisFinished();
        isFinished = true;
    }
}
