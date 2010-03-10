package sketch.ui.modifiers;

import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

/**
 * display a solution stack
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScSolutionStack extends ScModifierDispatcher {
    protected ScStack myStack;

    /**
     * @param stack
     *            already cloned stack
     */
    public ScSolutionStack(ScUiThread uiThread, ScUiList<ScModifierDispatcher> list,
            ScStack stack)
    {
        super(uiThread, list);
        myStack = stack;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    private class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            uiThread.gui.fillWithStack(ScSolutionStack.this, myStack);
        }
    }

    @Override
    public ScUiModifierInner getModifier() {
        return new Modifier();
    }

    @Override
    public String toString() {
        return "solution [cost=" + myStack.solutionCost + "] " + myStack.hashCode();
    }

    @Override
    public boolean isAcceptable() {
        return true;
    }
}
