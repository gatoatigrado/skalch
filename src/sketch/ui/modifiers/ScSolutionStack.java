package sketch.ui.modifiers;

import sketch.dyn.synth.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

public class ScSolutionStack extends ScModifierDispatcher {
    protected ScStack stack;

    public ScSolutionStack(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, ScStack stack)
    {
        super(uiThread, list);
        this.stack = stack;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    public class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.debugOutEditor.setText("solution with stack "
                    + stack.toString());
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }

    @Override
    public String toString() {
        return "solution " + stack.hashCode();
    }
}
