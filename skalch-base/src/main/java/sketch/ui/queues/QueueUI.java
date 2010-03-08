package sketch.ui.queues;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

public class QueueUI implements ScUserInterface {

    public final ScUserInterface base;
    private ScDynamicSketchCall<ScAngelicSketchBase> sketchCall;

    private Vector<Vector<Object>> listOfQueuesOutput;
    private String queueOutputFileName;
    private Queue previousQueues;
    private boolean isFinished;

    public QueueUI(ScUserInterface base,
            ScDynamicSketchCall<ScAngelicSketchBase> sketchCall,
            String queueOutputFileName, String queueInputFileName)
    {
        this.queueOutputFileName = queueOutputFileName;
        this.sketchCall = sketchCall;
        this.base = base;
        isFinished = false;

        if (queueInputFileName != "") {
            QueueFileInput input = new QueueFileInput(queueInputFileName);
            previousQueues = input.getQueue();
        }

        if (queueOutputFileName != "") {
            listOfQueuesOutput = new Vector<Vector<Object>>();
        }
    }

    public void addStackSolution(ScStack stack) {
        ScStack _stack = stack.clone();
        ScDebugRun debugRun = new ScDebugStackRun(sketchCall, _stack);
        debugRun.run();
        boolean isValid = true;

        if (previousQueues != null && !isFinished) {
            Vector<Object> queueTrace = debugRun.getQueueTrace();
            QueueIterator iterator = previousQueues.getIterator();
            for (int i = 0; i < queueTrace.size(); i++) {
                if (!iterator.checkValue(queueTrace.elementAt(i))) {
                    isValid = false;
                }
            }
            if (!iterator.canFinish()) {
                isValid = false;
            }
        }
        if (isValid) {
            if (listOfQueuesOutput != null && !isFinished) {
                Vector<Object> queue = debugRun.getQueue();
                listOfQueuesOutput.add(queue);
            }
            base.addStackSolution(stack);
        } else {
            // Used only for debugging purposes
            // DebugOut.print("Queues eliminated potential execution");
        }
    }

    public void addStackSynthesis(ScLocalStackSynthesis localSsr) {
        localSsr.queue = previousQueues;
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

    public void setCounterexamples(ScSolvingInputConf[] inputs) {
        base.setCounterexamples(inputs);
    }

    public void synthesisFinished() {
        if (listOfQueuesOutput != null) {
            try {
                ObjectOutputStream out =
                        new ObjectOutputStream(new FileOutputStream(
                                queueOutputFileName));
                out.writeObject(listOfQueuesOutput);
                out.close();
            } catch (FileNotFoundException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            } catch (IOException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            }
        }
        base.synthesisFinished();
        isFinished = true;
    }
}
