package sketch.result;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;

public class ScSynthesisResults {

    private Vector<ScStack> solutions;
    private Vector<ScLocalStackSynthesis> localSynths;

    private List<ScResultsObserver> observers;

    public ScSynthesisResults() {
        solutions = new Vector<ScStack>();
        localSynths = new Vector<ScLocalStackSynthesis>();
        observers = new ArrayList<ScResultsObserver>();
    }

    public void registerObserver(ScResultsObserver observer) {
        observers.add(observer);
    }

    public void detachObserver(ScResultsObserver observer) {
        observers.remove(observer);
    }

    public void addSynthesis(ScLocalStackSynthesis localSynth) {
        localSynths.add(localSynth);

        for (ScResultsObserver observer : observers) {
            observer.addStackSynthesis(localSynth);
        }
    }

    public void addStackSolution(ScStack stack) {
        solutions.add(stack);

        for (ScResultsObserver observer : observers) {
            observer.addStackSolution(stack);
        }
    }

    public boolean synthesisComplete() {
        // TODO Auto-generated method stub
        return false;
    }

    public void setSynthesisComplete() {

    }

    public void setCounterexamples(ScSolvingInputConf[] inputs) {
        for (ScResultsObserver observer : observers) {
            observer.setCounterexamples(inputs);
        }
    }

}
