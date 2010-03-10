package sketch.result;

import java.util.Vector;

import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.util.DebugOut;

public class ScSynthesisResults {

    private Vector<ScStack> solutions;
    private ScUserInterface ui;
    private Vector<ScLocalStackSynthesis> localSynths;

    public ScSynthesisResults(ScUserInterface ui) {
        this.ui = ui;
        ui.setScSynthesisResults(this);
        solutions = new Vector<ScStack>();
        localSynths = new Vector<ScLocalStackSynthesis>();
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSynth) {
        localSynths.add(localSynth);
    }

    public void addStackSolution(ScStack stack) {
        DebugOut.print(stack);
        solutions.add(stack);

        ui.addStackSolution(stack);
    }

    public boolean synthesisComplete() {
        // TODO Auto-generated method stub
        return false;
    }

    public void setSynthesisComplete() {

    }

}
