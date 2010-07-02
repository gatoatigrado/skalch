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

    final public ScStack myStack;

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

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((myStack == null) ? 0 : myStack.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        ScSolutionStack other = (ScSolutionStack) obj;
        if (myStack == null) {
            if (other.myStack != null) {
                return false;
            }
        } else if (!myStack.equals(other.myStack)) {
            return false;
        }
        return true;
    }
}
