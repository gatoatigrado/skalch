package sketch.result;

import java.util.List;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;

public interface ScResultsObserver {

    public void addStackSynthesis(ScLocalStackSynthesis localSynthesis);

    public void removeStackSynthesis(ScLocalStackSynthesis localSynthesis);

    public void removeAllSyntheses();

    public void addStackSolution(ScStack solution);

    public void removeStackSolution(ScStack solution);

    public void removeAllStackSolutions();

    public void resetStackSolutions(List<ScStack> solutions);

    public void setCounterexamples(ScSolvingInputConf[] inputs);
}
