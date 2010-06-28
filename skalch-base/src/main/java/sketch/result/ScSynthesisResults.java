package sketch.result;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;

public class ScSynthesisResults implements ScResultsObserver {

    // threadsafe
    private Vector<ScStack> solutions;
    private Vector<ScLocalStackSynthesis> localSynths;
    private boolean synthesisComplete;

    private List<ScResultsObserver> observers;

    public ScSynthesisResults() {
        solutions = new Vector<ScStack>();
        localSynths = new Vector<ScLocalStackSynthesis>();
        synthesisComplete = false;
        observers = new ArrayList<ScResultsObserver>();
    }

    public void registerObserver(ScResultsObserver observer) {
        observers.add(observer);
    }

    public void detachObserver(ScResultsObserver observer) {
        observers.remove(observer);
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSynth) {
        localSynths.add(localSynth);
        // localSynth.doneEvents.enqueue(this, "removeSynthesis", localSynth);

        for (ScResultsObserver observer : observers) {
            observer.addStackSynthesis(localSynth);
        }
    }

    public void removeStackSynthesis(ScLocalStackSynthesis localSynth) {
        localSynths.remove(localSynth);

        for (ScResultsObserver observer : observers) {
            observer.removeStackSynthesis(localSynth);
        }
    }

    public void removeAllSyntheses() {
        localSynths.removeAllElements();

        for (ScResultsObserver observer : observers) {
            observer.removeAllSyntheses();
        }
    }

    public void addStackSolution(ScStack stack) {
        ScStack clonedStack = stack.clone();
        solutions.add(clonedStack);

        for (ScResultsObserver observer : observers) {
            observer.addStackSolution(clonedStack);
        }
    }

    public void removeStackSolution(ScStack stack) {
        solutions.remove(stack);

        for (ScResultsObserver observer : observers) {
            observer.removeStackSolution(stack);
        }
    }

    public void removeAllStackSolutions() {
        solutions.removeAllElements();

        for (ScResultsObserver observer : observers) {
            observer.removeAllStackSolutions();
        }
    }

    public void resetStackSolutions(List<ScStack> newSolutions) {
        solutions.removeAllElements();
        solutions.addAll(newSolutions);

        for (ScResultsObserver observer : observers) {
            observer.resetStackSolutions(newSolutions);
        }
    }

    public boolean synthesisComplete() {
        return synthesisComplete;
    }

    public void setCounterexamples(ScSolvingInputConf[] inputs) {
        for (ScResultsObserver observer : observers) {
            observer.setCounterexamples(inputs);
        }
    }

    public ArrayList<ScStack> getSolutions() {
        return new ArrayList<ScStack>(solutions);
    }

    public void synthesisFinished() {
        synthesisComplete = true;
        for (ScResultsObserver observer : observers) {
            observer.synthesisFinished();
        }
    }
}
