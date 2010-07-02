package sketch.ui.gui;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Vector;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.SwingUtilities;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.stats.ScStatsModifier;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.result.ScSynthesisResults;
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
    public ScDynamicSketchCall<?> sketchCall;
    public ScUiGui gui;
    public ScFixedInputConf[] allCounterexamples;
    public AtomicInteger modifierTimestamp = new AtomicInteger(0);
    static ConcurrentLinkedQueue<ScUiModifier> modifierList =
            new ConcurrentLinkedQueue<ScUiModifier>();
    public boolean autoDisplayFirstSolution = true;
    public BackendOptions beOpts;
    public ScModifierDispatcher lastDisplayDispatcher;
    public Vector<ScSourceConstruct> sourceCodeInfo;
    public ScSynthesisResults results;

    public ScUiThread(ScSynthesisResults results, ScDynamicSketchCall<?> sketchCall,
            BackendOptions beOpts, Set<ScSourceConstruct> sourceCodeInfo)
    {
        super(0.05f);
        this.sketchCall = sketchCall;
        this.beOpts = beOpts;
        this.sourceCodeInfo = new Vector<ScSourceConstruct>();
        this.sourceCodeInfo.addAll(sourceCodeInfo);
        this.results = results;
        results.registerObserver(this);
        autoDisplayFirstSolution = !beOpts.uiOpts.noAutoSolnDisp;
    }

    @Override
    public void init() {
        gui = new ScUiGui(this);
        gui.setVisible(true);
        new AnimatedRunnableModifier(1.f) {
            @Override
            public void run() {
                ScStatsMT.statsSingleton.showStatsWithUi();
            }
        };
    }

    @Override
    public void run_inner() {
        int numModifiers = modifierList.size();
        for (int a = 0; a < numModifiers; a++) {
            ScUiModifier m = modifierList.remove();
            GuiThreadTask runModifier = new GuiThreadTask(m);
            SwingUtilities.invokeLater(runModifier);
        }
    }

    @Override
    public void finish() {
        gui.setVisible(false);
        gui.dispose();
    }

    public void modifierComplete(ScUiModifier m) {
        modifierList.add(m);
    }

    public int nextModifierTimestamp() {
        return modifierTimestamp.incrementAndGet();
    }

    public void addStackSynthesis(final ScLocalStackSynthesis localSsr) {
        final ScUiThread target = this;
        new AddedModifier() {
            @Override
            public void apply() {
                ScUiGui gui = target.gui;
                gui.numSynthActive += 1;
                new ScActiveStack(target, gui.synthCompletions, localSsr).add();
            }
        };
    }

    public void removeAllSyntheses() {
        final ScUiThread target = this;
        new RemoveModifier() {
            @Override
            public void apply() {
                ScUiGui gui = target.gui;
                Object[] completions = gui.synthCompletions.getAll();
                for (Object completion : completions) {
                    if (completion instanceof ScActiveStack) {
                        ScActiveStack stack = (ScActiveStack) completion;
                        gui.synthCompletions.remove(stack);
                    }
                }
            }
        };
    }

    public void removeStackSynthesis(final ScLocalStackSynthesis localSsr) {
        final ScUiThread target = this;
        new RemoveModifier() {
            @Override
            public void apply() {
                ScUiGui gui = target.gui;
                Object[] completions = gui.synthCompletions.getAll();
                for (Object completion : completions) {
                    if (completion instanceof ScActiveStack) {
                        ScActiveStack stack = (ScActiveStack) completion;
                        if (stack.localSsr == localSsr) {
                            gui.synthCompletions.remove(stack);
                        }
                    }
                }
            }
        };
    }

    public void addStackSolution(final ScStack stack) {
        new AddedModifier() {
            @Override
            public void apply() {
                ScSolutionStack solution =
                        new ScSolutionStack(ScUiThread.this, gui.synthCompletions, stack);
                solution.add();
                autoDisplaySolution(solution);
            }
        };
    }

    public void removeAllStackSolutions() {
        final ScUiThread target = this;
        new RemoveModifier() {
            @Override
            public void apply() {
                ScUiGui gui = target.gui;
                Object[] completions = gui.synthCompletions.getAll();
                for (Object completion : completions) {
                    if (completion instanceof ScSolutionStack) {
                        ScSolutionStack solution = (ScSolutionStack) completion;
                        gui.synthCompletions.remove(solution);
                    }
                }
            }
        };
    }

    public void removeStackSolution(final ScStack stack) {
        new RemoveModifier() {
            @Override
            public void apply() {
                ScUiGui gui = ScUiThread.this.gui;
                Object[] completions = gui.synthCompletions.getAll();
                for (Object completion : completions) {
                    if (completion instanceof ScSolutionStack) {
                        ScSolutionStack solution = (ScSolutionStack) completion;
                        if (stack == solution.myStack) {
                            gui.synthCompletions.remove(solution);
                        }
                    }
                }
            }
        };
    }

    public void resetStackSolutions(final List<ScStack> stacks) {
        final ScUiThread target = this;
        new RemoveModifier() {
            @Override
            // TODO (sbarman): this is a hack and needs to be rewritten
            public void apply() {
                ScUiGui gui = target.gui;
                gui.synthCompletions.removeAll();
                // Object[] completions = gui.synthCompletions.getAll();
                // for (Object completion : completions) {
                // if (completion instanceof ScSolutionStack) {
                // ScSolutionStack solution = (ScSolutionStack) completion;
                // gui.synthCompletions.remove(solution);
                // }
                // }
                List<ScSolutionStack> solutions = new ArrayList<ScSolutionStack>();
                for (ScStack stack : stacks) {
                    ScSolutionStack solution =
                            new ScSolutionStack(ScUiThread.this, gui.synthCompletions,
                                    stack);
                    solutions.add(solution);
                }
                gui.synthCompletions.addAll(solutions);
            }
        };
    }

    /**
     * Currently usused, as we don't do much with counterexamples.
     */
    public void setCounterexamples(ScSolvingInputConf[] inputs) {
        if (beOpts.uiOpts.printCounterex) {
            Object[] text = { "counterexamples", inputs };
            DebugOut.print_colored(DebugOut.BASH_GREEN, "[user requested print]", "\n",
                    true, text);
        }
        allCounterexamples = ScFixedInputConf.fromInputs(inputs);
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
        if (autoDisplayFirstSolution) {
            autoDisplayFirstSolution = false;
            gui.synthCompletions.setSelected(solution);
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

    private abstract class RemoveModifier extends ScUiModifierInner {
        public RemoveModifier() {
            remove();
        }

        protected void remove() {
            try {
                new ScUiModifier(ScUiThread.this, this).enqueueTo();
            } catch (ScUiQueueableInactive e) {
                e.printStackTrace();
            }
        }
    }

    public abstract class AnimatedRunnableModifier extends AddedModifier {
        public float timeoutSecs;
        public float lastTime = 0.f;

        public AnimatedRunnableModifier(float timeoutSecs) {
            this.timeoutSecs = timeoutSecs;
        }

        @Override
        public void apply() {
            if (!results.synthesisComplete()) {
                add(); // re-enqueue;
            }
            if (thread_time - lastTime > timeoutSecs) {
                run();
                lastTime = thread_time;
            }
        }

        public abstract void run();
    }

    public void synthesisFinished() {
        DebugOut.print("Synthesis finished");
        ScStatsMT.statsSingleton.showStatsWithUi();
    }

}
