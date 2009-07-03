package sketch.ui.modifiers;

import sketch.dyn.synth.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

public class ScSolutionStack extends ScModifierDispatcher {
    protected ScStack my_stack;

    /**
     * @param stack
     *            already cloned stack
     */
    public ScSolutionStack(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, ScStack stack)
    {
        super(uiThread, list);
        my_stack = stack;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    public class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.fillWithStack(my_stack);
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }

    @Override
    public int getCost() {
        return my_stack.solution_cost;
    }

    @Override
    public String toString() {
        return "solution [cost=" + my_stack.solution_cost + "] "
                + my_stack.hashCode();
    }

    @Override
    public boolean isAcceptable() {
        return true;
    }
}
