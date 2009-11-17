package sketch.ui.trace;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Vector;

import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

public class TraceUI implements ScUserInterface {

    public final ScUserInterface base;
    private String trace_output_file_name;

    private Vector<String[]> listOfTracesOutput;
    private boolean isFinished;

    public TraceUI(ScUserInterface base, String trace_output_file_name) {
        this.trace_output_file_name = trace_output_file_name;
        this.base = base;
        isFinished = false;

        if (trace_output_file_name != "") {
            listOfTracesOutput = new Vector<String[]>();
        }
    }

    public void addGaSolution(ScGaIndividual individual) {
        base.addGaSolution(individual);
    }

    public void addGaSynthesis(ScGaSynthesis scGaSynthesis) {
        base.addGaSynthesis(scGaSynthesis);
    }

    public void addStackSolution(ScStack stack) {
        ScStack _stack = stack.clone();

        if (listOfTracesOutput != null && !isFinished) {
            String[] trace = _stack.getStringArrayRep();
            listOfTracesOutput.add(trace);
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
        if (listOfTracesOutput != null) {
            try {
                PrintStream out = new PrintStream(new FileOutputStream(
                        trace_output_file_name));
                for (String[] trace : listOfTracesOutput) {
                    for (int i = 0; i < trace.length; i++) {
                        out.print(trace[i]);
                        if (i != trace.length - 1) {
                            out.print(";");
                        }
                    }
                    out.println();
                }
                out.close();
            } catch (FileNotFoundException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            } catch (IOException e) {
                DebugOut.print_exception("Problem opening file for queues", e);
            }
            base.synthesisFinished();
        }
        isFinished = true;
    }
}
