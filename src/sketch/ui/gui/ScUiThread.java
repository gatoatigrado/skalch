package sketch.ui.gui;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.SwingUtilities;

import sketch.dyn.BackendOptions;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;
import sketch.ui.modifiers.ScActiveStack;
import sketch.ui.modifiers.ScSolutionStack;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.modifiers.ScUiModifierInner;
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
    public ScStackSynthesis ssr;
    public ScDynamicSketch sketch;
    public ScUiGui gui;
    public AtomicInteger modifier_timestamp = new AtomicInteger(0);
    static ConcurrentLinkedQueue<ScUiThread> gui_list =
            new ConcurrentLinkedQueue<ScUiThread>();
    static ConcurrentLinkedQueue<ScUiModifier> modifier_list =
            new ConcurrentLinkedQueue<ScUiModifier>();
    public boolean auto_display_first_solution = true;

    public ScUiThread(ScStackSynthesis ssr, ScDynamicSketch sketch) {
        super(0.1f);
        this.ssr = ssr;
        this.sketch = sketch;
        auto_display_first_solution =
                !BackendOptions.ui_opts.bool_("no_auto_soln_disp");
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
                new ScActiveStack(target, gui.synthCompletions, local_ssr)
                        .add();
            }
        }).add();
    }

    public void addSolution(ScStack stack) {
        final ScStack stack_to_add = stack.clone();
        final ScUiThread target = this;
        new RunnableModifier(new Runnable() {
            public void run() {
                ScSolutionStack solution =
                        new ScSolutionStack(target,
                                target.gui.synthCompletions, stack_to_add);
                solution.add();
                if (auto_display_first_solution) {
                    auto_display_first_solution = false;
                    solution.dispatch();
                }
            }
        }).add();
    }

    public void set_counterexamples(ScSolvingInputConf[] inputs) {
        final ScFixedInputConf[] counterexamples =
                ScFixedInputConf.from_inputs(inputs);
        final ScUiThread target = this;
        new RunnableModifier(new Runnable() {
            public void run() {
                for (ScFixedInputConf elt : counterexamples) {
                    target.gui.inputChoices.add(elt);
                }
            }
        }).add();
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
