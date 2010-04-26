package sketch.ui.modifiers;

import java.util.List;

import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.Trace;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

public class ScAngelicEntangledStack extends ScModifierDispatcher {
    final public ScStack myStack;
    private List<DynAngel> entangledSubset;

    /**
     * @param stack
     *            already cloned stack
     */
    public ScAngelicEntangledStack(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, Trace trace,
            List<DynAngel> entangledSubset)
    {
        super(uiThread, list);
        this.entangledSubset = entangledSubset;
        List<Event> events = trace.getEvents();
        for (Event event : events) {

        }
        myStack = null;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    private class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            uiThread.gui.fillWithStack(ScAngelicEntangledStack.this, myStack);
        }
    }

    @Override
    public ScUiModifierInner getModifier() {
        return new Modifier();
    }

    @Override
    public String toString() {
        return "angelic subset [" + entangledSubset + "]";
    }

    @Override
    public boolean isAcceptable() {
        return true;
    }
}
