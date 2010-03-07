package sketch.ui.gui;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.SwingUtilities;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScActiveStack;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.modifiers.ScSolutionStack;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.modifiers.ScUiModifierInner;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.thread.InteractiveThread;

/**
 * Thread which launches the user interface and shuts it down when requested. Perhaps
 * slightly unnecessary but nice for code organization.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUiThread extends InteractiveThread implements ScUserInterface {
    public ScSynthesis<?> synth_runtime;
    public ScDynamicSketchCall<?> sketch_call;
    public ScUiGui gui;
    public ScFixedInputConf[] all_counterexamples;
    public AtomicInteger modifier_timestamp = new AtomicInteger(0);
    static ConcurrentLinkedQueue<ScUiThread> gui_list =
            new ConcurrentLinkedQueue<ScUiThread>();
    static ConcurrentLinkedQueue<ScUiModifier> modifier_list =
            new ConcurrentLinkedQueue<ScUiModifier>();
    public boolean auto_display_first_solution = true;
    public BackendOptions be_opts;
    public ScModifierDispatcher lastDisplayDispatcher;
    private ScSourceConstruct sourceInfo;

    public ScUiThread(ScSynthesis<?> synth_runtime, ScDynamicSketchCall<?> sketch_call,
            BackendOptions be_opts, ScSourceConstruct sourceInfo)
    {
        super(0.05f);
        this.synth_runtime = synth_runtime;
        this.sketch_call = sketch_call;
        this.be_opts = be_opts;
        this.sourceInfo = sourceInfo;
        auto_display_first_solution = !be_opts.ui_opts.no_auto_soln_disp;
        gui_list.add(this);
    }

    @Override
    public void init() {
        gui = new ScUiGui(this);
        gui.setVisible(true);
        new AnimatedRunnableModifier(1.f) {
            @Override
            public void run() {
                ScStatsMT.stats_singleton.showStatsWithUi();
            }
        };
    }

    @Override
    public void run_inner() {
        int num_modifiers = modifier_list.size();
        for (int a = 0; a < num_modifiers; a++) {
            ScUiModifier m = modifier_list.remove();
            GuiThreadTask run_modifier = new GuiThreadTask(m);
            SwingUtilities.invokeLater(run_modifier);
        }
    }

    @Override
    public void finish() {
        gui.setVisible(false);
        gui.dispose();
    }

    public void modifierComplete(ScUiModifier m) {
        modifier_list.add(m);
    }

    public int nextModifierTimestamp() {
        return modifier_timestamp.incrementAndGet();
    }

    public void addStackSynthesis(final ScLocalStackSynthesis local_ssr) {
        final ScUiThread target = this;
        new AddedModifier() {
            @Override
            public void apply() {
                ScUiGui gui = target.gui;
                gui.num_synth_active += 1;
                new ScActiveStack(target, gui.synthCompletions, local_ssr).add();
            }
        };
    }

    public void addStackSolution(ScStack stack__) {
        final ScStack stack_to_add = stack__.clone();
        new AddedModifier() {
            @Override
            public void apply() {
                ScSolutionStack solution =
                        new ScSolutionStack(ScUiThread.this, gui.synthCompletions,
                                stack_to_add);
                solution.add();
                autoDisplaySolution(solution);
            }
        };
    }

    /**
     * Currently usused, as we don't do much with counterexamples.
     */
    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (be_opts.ui_opts.print_counterex) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[user requested print]", "\n",
                    true, text);
        }
        all_counterexamples = ScFixedInputConf.from_inputs(inputs);
    }

    /**
     * a.t.m. more of a demonstration of how this can be flexible; the code doesn't need
     * to be this verbose.
     */
    public void setStats(final ScStatsModifier modifier) {
        ScGuiStatsWarningsPrinter wp = new ScGuiStatsWarningsPrinter();
        modifier.execute(wp);
        ScGuiStatsEntriesPrinter ep = new ScGuiStatsEntriesPrinter(wp, this);
        modifier.execute(ep);
        ep.dispatch();
    }

    /** dispatch a solution modifier if nothing has been displayed already. */
    protected void autoDisplaySolution(ScModifierDispatcher solution) {
        if (auto_display_first_solution) {
            auto_display_first_solution = false;
            gui.synthCompletions.set_selected(solution);
            solution.dispatch();
        }
    }

    // all of this junk just because Java can't bind non-final variables
    public class GuiThreadTask implements Runnable {
        protected ScUiModifier m;

        public GuiThreadTask(ScUiModifier m) {
            this.m = m;
        }

        public void run() {
            m.modifier.apply();
        }
    }

    private abstract class AddedModifier extends ScUiModifierInner {
        public AddedModifier() {
            add();
        }

        protected void add() {
            try {
                new ScUiModifier(ScUiThread.this, this).enqueueTo();
            } catch (ScUiQueueableInactive e) {
                e.printStackTrace();
            }
        }
    }

    public abstract class AnimatedRunnableModifier extends AddedModifier {
        public float timeout_secs;
        public float last_time = 0.f;

        public AnimatedRunnableModifier(float timeout_secs) {
            this.timeout_secs = timeout_secs;
        }

        @Override
        public void apply() {
            if (!synth_runtime.wait_handler.synthesis_complete.get()) {
                add(); // re-enqueue;
            }
            if (thread_time - last_time > timeout_secs) {
                run();
                last_time = thread_time;
            }
        }

        public abstract void run();
    }

    public void synthesisFinished() {
        DebugOut.print("Synthesis finished");
    }
}
