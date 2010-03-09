package sketch.result;

import java.util.Vector;

import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.util.DebugOut;

public class ScSynthesisResults {

    Vector<ScStack> solutions;
    private ScUserInterface ui;

    public ScSynthesisResults(ScUserInterface ui) {
        this.ui = ui;
        solutions = new Vector<ScStack>();
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSynth) {
    // TODO Auto-generated method stub

    }

    public void addStackSolution(ScStack stack) {
        DebugOut.print(stack);
        solutions.add(stack);

        ui.addStackSolution(stack);
    }

}
