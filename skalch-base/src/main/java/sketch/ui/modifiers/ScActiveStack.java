package sketch.ui.modifiers;

import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

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

    private class SynthDoneModifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.num_synth_active -= 1;
            if (ui_thread.gui.num_synth_active <= 0) {
                ui_thread.gui.disableStopButton();
            }
            list.remove(ScActiveStack.this);
            list.add(new ScLongestStack(ScActiveStack.this));
            list.add(new ScRandomStack(ScActiveStack.this));
        }
    }

    public void synthDone() {
        try {
            new ScUiModifier(ui_thread, new SynthDoneModifier()).enqueueTo();
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
        }
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
                ScLocalStackSynthesis.ScStackSynthesisThread synthThread,
                ScStack stack)
        {
            my_stack = stack.clone();
        }

        @Override
        public void apply() {
            ui_thread.auto_display_first_solution = false;
            ui_thread.gui.fillWithStack(ScActiveStack.this, my_stack);
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }
}
