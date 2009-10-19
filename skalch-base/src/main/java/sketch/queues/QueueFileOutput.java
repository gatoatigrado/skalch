package sketch.queues;

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
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

public class QueueFileOutput implements ScUserInterface {

	public final ScUserInterface base;
	private ScDynamicSketchCall<ScAngelicSketchBase> sketch_call;
	private Vector<Vector<Object>> listOfQueues;
	private ObjectOutputStream out;

	public QueueFileOutput(ScUserInterface base,
			ScDynamicSketchCall<ScAngelicSketchBase> sketch_call,
			String file_name) {
		this.sketch_call = sketch_call;
		this.base = base;
		listOfQueues = new Vector<Vector<Object>>();
		try {
			out = new ObjectOutputStream(new FileOutputStream(file_name));
		} catch (FileNotFoundException e) {
			DebugOut.print_exception("Problem opening file for queues", e);
		} catch (IOException e) {
			DebugOut.print_exception("Problem opening file for queues", e);
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
		ScDebugRun debugRun = new ScDebugStackRun(sketch_call, _stack);
		debugRun.run();
		Vector<Object> queue = debugRun.get_queue();
		if (out != null) {
			try {
				out.writeObject(queue);
				out.flush();
			} catch (IOException e) {
				DebugOut.print_exception("Problem writing out queues", e);
			}
		}
		// Used only for debugging purposes
		listOfQueues.add(queue);
		base.addStackSolution(stack);
	}

	public void addStackSynthesis(ScLocalStackSynthesis localSsr) {
		localSsr.done_events.enqueue(this, "localStackSynthesisFinished");
		base.addStackSynthesis(localSsr);
	}

	public void localStackSynthesisFinished() {
		/*
		 * if (out != null) { try { out.flush(); } catch (IOException e) {
		 * DebugOut.print_exception("Problem writing out queues", e); } }
		 */
		System.out.println("");
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

}
