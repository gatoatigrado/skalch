package sketch.ui;

import javax.swing.event.ListSelectionEvent;

import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.dyn.synth.ScLocalStackSynthesis.SynthesisThread;
import sketch.util.DebugOut;

/**
 * kinda because I can't resist using Eclipse's formatter (it also separates
 * generated code)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiGui extends gui_0_1 {
    private static final long serialVersionUID = 6584583626375432139L;
    public ScUiThread ui_thread;
    protected ScUiList<ScCounterexample> inputChoices;
    protected ScUiList<ScUiModifierDispatcher> synthCompletions;

    public ScUiGui(ScUiThread ui_thread) {
        super();
        this.ui_thread = ui_thread;
        inputChoices = new ScUiList<ScCounterexample>(selectInputList);
        synthCompletions =
                new ScUiList<ScUiModifierDispatcher>(synthCompletionList);
    }

    public void addStackSynthesis(ScLocalStackSynthesis local_ssr) {
        synthCompletions.add(new StackModifierDispatcher(local_ssr));
    }

    public class StackModifierDispatcher extends ScUiModifierDispatcher {
        public ScLocalStackSynthesis local_ssr;

        public StackModifierDispatcher(ScLocalStackSynthesis local_ssr) {
            this.local_ssr = local_ssr;
        }

        private class Modifier extends ScUiModifier {
            // ScCounterexample[] counterexamples;
            ScStack my_stack;

            public Modifier() {
                super(ui_thread);
                // counterexamples = inputChoices.getSelected();
            }

            @Override
            public void setInfoInner(ScLocalStackSynthesis localSynth,
                    SynthesisThread synthThread, ScStack stack)
            {
                my_stack = stack.clone();
            }

            @Override
            public void apply() {
                DebugOut.print("apply...");
                debugOutEditor.setText("hello world from StackModifier;"
                        + " stack = " + my_stack.toString());
            }
        }

        @Override
        public String toString() {
            return "current completions for stack synthesis " + local_ssr.uid;
        }

        @Override
        public void dispatch() {
            (new Modifier()).enqueueTo(local_ssr);
        }
    }

    @Override
    protected void synthCompletionSelectionChanged(ListSelectionEvent evt) {
        ScUiModifierDispatcher[] selected = synthCompletions.getSelected();
        viewSelectionsButton.setEnabled(selected.length == 1);
    }

    @Override
    protected void viewSelections() {
        synthCompletions.getSelected()[0].dispatch();
    }
}
