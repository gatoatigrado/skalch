package sketch.ui.queues;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.result.ScResultsObserver;
import sketch.util.DebugOut;

public class QueueOutput implements ScResultsObserver {

    private ScDynamicSketchCall<ScAngelicSketchBase> sketchCall;

    private Vector<Vector<Object>> listOfQueuesOutput;
    private String queueOutputFileName;
    private boolean isFinished;

    public QueueOutput(ScDynamicSketchCall<ScAngelicSketchBase> sketchCall,
            String queueOutputFileName)
    {
        this.queueOutputFileName = queueOutputFileName;
        this.sketchCall = sketchCall;
        isFinished = false;

        if (queueOutputFileName != "") {
            listOfQueuesOutput = new Vector<Vector<Object>>();
        }
    }

    public void addStackSolution(ScStack stack) {
        ScStack _stack = stack.clone();
        ScDebugRun debugRun = new ScDebugStackRun(sketchCall, _stack);
        debugRun.run();
        if (listOfQueuesOutput != null && !isFinished) {
            Vector<Object> queue = debugRun.getQueue();
            listOfQueuesOutput.add(queue);
        }
    }

    public void synthesisFinished() {
        if (listOfQueuesOutput != null) {
            try {
                ObjectOutputStream out =
                        new ObjectOutputStream(new FileOutputStream(queueOutputFileName));
                out.writeObject(listOfQueuesOutput);
                out.close();
            } catch (FileNotFoundException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            } catch (IOException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            }
        }
        isFinished = true;
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSynthesis) {
    // Do nothing
    }

    public void removeAllStackSolutions() {
    // Do nothing
    }

    public void removeAllSyntheses() {
    // Do nothing
    }

    public void removeStackSolution(ScStack solution) {
    // Do nothing
    }

    public void removeStackSynthesis(ScLocalStackSynthesis localSynthesis) {
    // Do nothing
    }

    public void resetStackSolutions(List<ScStack> solutions) {
    // Do nothing
    }

    public void setCounterexamples(ScSolvingInputConf[] inputs) {
    // Do nothing
    }
}
