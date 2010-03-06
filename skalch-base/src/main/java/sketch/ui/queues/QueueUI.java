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
    private ScDynamicSketchCall<ScAngelicSketchBase> sketch_call;

    private Vector<Vector<Object>> listOfQueuesOutput;
    private String queue_output_file_name;
    private Queue previousQueues;
    private boolean isFinished;

    public QueueUI(ScUserInterface base,
            ScDynamicSketchCall<ScAngelicSketchBase> sketch_call,
            String queue_output_file_name, String queue_input_file_name)
    {
        this.queue_output_file_name = queue_output_file_name;
        this.sketch_call = sketch_call;
        this.base = base;
        isFinished = false;

        if (queue_input_file_name != "") {
            QueueFileInput input = new QueueFileInput(queue_input_file_name);
            previousQueues = input.getQueue();
        }

        if (queue_output_file_name != "") {
            listOfQueuesOutput = new Vector<Vector<Object>>();
        }
    }

    public void addStackSolution(ScStack stack) {
        ScStack _stack = stack.clone();
        ScDebugRun debugRun = new ScDebugStackRun(sketch_call, _stack);
        debugRun.run();
        boolean isValid = true;

        if (previousQueues != null && !isFinished) {
            Vector<Object> queue_trace = debugRun.get_queue_trace();
            QueueIterator iterator = previousQueues.getIterator();
            for (int i = 0; i < queue_trace.size(); i++) {
                if (!iterator.checkValue(queue_trace.elementAt(i))) {
                    isValid = false;
                }
            }
            if (!iterator.canFinish()) {
                isValid = false;
            }
        }
        if (isValid) {
            if (listOfQueuesOutput != null && !isFinished) {
                Vector<Object> queue = debugRun.get_queue();
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

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        base.set_counterexamples(inputs);
    }

    public void synthesisFinished() {
        if (listOfQueuesOutput != null) {
            try {
                ObjectOutputStream out =
                        new ObjectOutputStream(new FileOutputStream(
                                queue_output_file_name));
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
