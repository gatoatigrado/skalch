package sketch.ui.modifiers;

import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

/**
 * UI Modifier dispatcher for an active stack.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScActiveStack extends ScLocalSynthDispatcher {
    public ScActiveStack(ScUiThread uiThread, ScUiList<ScModifierDispatcher> list,
            ScLocalStackSynthesis localSsr)
    {
        super(uiThread, list, localSsr);
        this.localSsr.doneEvents.enqueue(this, "synthDone");
    }

    private class SynthDoneModifier extends ScUiModifierInner {
        @Override
        public void apply() {
            uiThread.gui.numSynthActive -= 1;
            if (uiThread.gui.numSynthActive <= 0) {
                uiThread.gui.disableStopButton();
            }
            list.remove(ScActiveStack.this);
            list.add(new ScLongestStack(ScActiveStack.this));
            list.add(new ScRandomStack(ScActiveStack.this));
        }
    }

    public void synthDone() {
        try {
            new ScUiModifier(uiThread, new SynthDoneModifier()).enqueueTo();
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
        }
    }

    @Override
    public String toString() {
        return "current completions for stack synthesis " + localSsr.uid;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo(localSsr);
    }

    // === modifier ===
    private class Modifier extends ScUiModifierInner {
        // ScCounterexample[] counterexamples;
        ScStack myStack;

        @Override
        public void setInfo(ScLocalStackSynthesis localSynth,
                ScLocalStackSynthesis.ScStackSynthesisThread synthThread, ScStack stack)
        {
            myStack = stack.clone();
        }

        @Override
        public void apply() {
            uiThread.autoDisplayFirstSolution = false;
            uiThread.gui.fillWithStack(ScActiveStack.this, myStack);
        }
    }

    @Override
    public ScUiModifierInner getModifier() {
        return new Modifier();
    }
}
