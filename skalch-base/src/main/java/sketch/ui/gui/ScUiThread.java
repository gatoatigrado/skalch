package sketch.ui.gui;

import static sketch.dyn.BackendOptions.beopts;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.SwingUtilities;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.ctrls.ScGaCtrlConf;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScGaInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScActiveGaDispatcher;
import sketch.ui.modifiers.ScActiveStack;
import sketch.ui.modifiers.ScGaSolutionDispatcher;
import sketch.ui.modifiers.ScSolutionStack;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.modifiers.ScUiModifierInner;
import sketch.util.DebugOut;
import sketch.util.InteractiveThread;

/**
 * Thread which launches the user interface and shuts it down when requested.
 * Perhaps slightly unnecessary but nice for code organization.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiThread extends InteractiveThread implements ScUserInterface {
    public ScSynthesis<?> synth_runtime;
    public ScDynamicSketchCall<?> sketch_call;
    public ScGaCtrlConf ga_ctrl_conf;
    public ScGaInputConf ga_oracle_conf;
    public ScUiGui gui;
    public ScFixedInputConf[] all_counterexamples;
    public AtomicInteger modifier_timestamp = new AtomicInteger(0);
    static ConcurrentLinkedQueue<ScUiThread> gui_list =
            new ConcurrentLinkedQueue<ScUiThread>();
    static ConcurrentLinkedQueue<ScUiModifier> modifier_list =
            new ConcurrentLinkedQueue<ScUiModifier>();
    public boolean auto_display_first_solution = true;
    public BackendOptions be_opts;

    public ScUiThread(ScSynthesis<?> synth_runtime,
            ScDynamicSketchCall<?> sketch_call, BackendOptions be_opts)
    {
        super(0.05f);
        this.synth_runtime = synth_runtime;
        this.sketch_call = sketch_call;
        this.be_opts = be_opts;
        if (be_opts.synth_opts.solver.isGa) {
            ga_ctrl_conf = new ScGaCtrlConf();
            ga_oracle_conf = new ScGaInputConf();
        }
        auto_display_first_solution = !be_opts.ui_opts.no_auto_soln_disp;
        gui_list.add(this);
    }

    @Override
    public void init() {
        gui = new ScUiGui(this);
        gui.setVisible(true);
    }

    @Override
    public void run_inner() {
        try {
            while (!modifier_list.isEmpty()) {
                ScUiModifier m = modifier_list.remove();
                RunModifier run_modifier = new RunModifier(m);
                SwingUtilities.invokeLater(run_modifier);
            }
        } catch (NoSuchElementException e) {
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
        new RunnableModifier(new Runnable() {
            public void run() {
                ScUiGui gui = target.gui;
                gui.num_synth_active += 1;
                new ScActiveStack(target, gui.synthCompletions, local_ssr)
                        .add();
            }
        }).add();
    }

    public void addStackSolution(ScStack stack__) {
        final ScStack stack_to_add = stack__.clone();
        new RunnableModifier(new Runnable() {
            public void run() {
                ScSolutionStack solution =
                        new ScSolutionStack(ScUiThread.this,
                                gui.synthCompletions, stack_to_add);
                solution.add();
                if (auto_display_first_solution) {
                    auto_display_first_solution = false;
                    gui.synthCompletions.set_selected(solution);
                    solution.dispatch();
                }
            }
        }).add();
    }

    public void addGaSynthesis(final ScGaSynthesis sc_ga_synthesis) {
        new RunnableModifier(new Runnable() {
            public void run() {
                gui.num_synth_active += 1;
                new ScActiveGaDispatcher(ScUiThread.this, gui.synthCompletions,
                        sc_ga_synthesis).add();
            }
        }).add();
    }

    public void addGaSolution(ScGaIndividual individual__) {
        final ScGaIndividual individual = individual__.clone();
        new RunnableModifier(new Runnable() {
            public void run() {
                ScGaSolutionDispatcher solution_individual =
                        new ScGaSolutionDispatcher(individual, ScUiThread.this,
                                gui.synthCompletions);
                solution_individual.add();
                if (auto_display_first_solution) {
                    auto_display_first_solution = false;
                    solution_individual.dispatch();
                }
            }
        }).add();
    }

    public void displayAnimated(ScGaIndividual individual__) {
        final ScGaIndividual individual = individual__.clone();
        new RunnableModifier(new Runnable() {
            public void run() {
                ScGaSolutionDispatcher solution_individual =
                        new ScGaSolutionDispatcher(individual, ScUiThread.this,
                                gui.synthCompletions);
                solution_individual.dispatch();
            }
        }).add();
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        if (beopts().ui_opts.print_counterexamples) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN,
                    "[user requested print]", "\n", true, text);
        }
        all_counterexamples = ScFixedInputConf.from_inputs(inputs);
    }

    // all of this junk just because Java can't bind non-final variables
    public class RunModifier implements Runnable {
        protected ScUiModifier m;

        public RunModifier(ScUiModifier m) {
            this.m = m;
        }

        public void run() {
            m.modifier.apply();
        }
    }

    public class RunnableModifier extends ScUiModifierInner {
        private Runnable runnable;

        public RunnableModifier(Runnable runnable) {
            this.runnable = runnable;
        }

        public void add() {
            try {
                new ScUiModifier(ScUiThread.this, this).enqueueTo();
            } catch (ScUiQueueableInactive e) {
                e.printStackTrace();
            }
        }

        @Override
        public void apply() {
            runnable.run();
        }
    }
}
