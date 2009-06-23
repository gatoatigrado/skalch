package sketch.ui.modifiers;

import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.dyn.synth.ScLocalStackSynthesis.SynthesisThread;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUiThread;

/**
 * UI Modifier dispatcher for an active stack.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScActiveStack extends ScLocalSynthDispatcher {

    public ScActiveStack(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, ScLocalStackSynthesis localSsr)
    {
        super(uiThread, list, localSsr);
        local_ssr.done_events.enqueue(this, "synthDone");
    }

    public void synthDone() {
        list.remove(this);
        list.add(new ScLongestStack(this));
        list.add(new ScRandomStack(this));
    }

    @Override
    public String toString() {
        return "current completions for stack synthesis " + local_ssr.uid;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo(local_ssr);
    }

    // === modifier ===

    private class Modifier extends ScUiModifierInner {
        // ScCounterexample[] counterexamples;
        ScStack my_stack;

        @Override
        public void setInfo(ScLocalStackSynthesis localSynth,
                SynthesisThread synthThread, ScStack stack)
        {
            my_stack = stack.clone();
        }

        @Override
        public void apply() {
            ui_thread.gui.debugOutEditor.setText("hello world from "
                    + "StackModifier; stack = " + my_stack.toString());
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }
}
