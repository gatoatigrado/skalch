package sketch.ui.modifiers;

import static sketch.util.DebugOut.assertFalse;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

/**
 * A class which has bound variables. This is useful for the synthesis list, where
 * different synthesizers may have significantly different functionalities for showing
 * in-progress synthesis.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScModifierDispatcher {
    public ScUiThread uiThread;
    public ScUiList<ScModifierDispatcher> list;

    public ScModifierDispatcher(ScUiThread uiThread, ScUiList<ScModifierDispatcher> list)
    {
        this.uiThread = uiThread;
        this.list = list;
    }

    public void add() {
        if (list == null) {
            assertFalse("ScModifierDispatcher.add() -- list is null");
        }
        list.add(this);
    }

    public void dispatch() {
        try {
            ScUiModifierInner modifierInner = getModifier();
            if (modifierInner == null) {
                assertFalse("class", this.getClass().getName(), "returned null modifier");
            }
            enqueue(new ScUiModifier(uiThread, modifierInner));
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
            if (list != null) {
                list.remove(this);
            }
        }
    }

    public abstract ScUiModifierInner getModifier();

    @Override
    public abstract String toString();

    public abstract void enqueue(ScUiModifier m) throws ScUiQueueableInactive;

    public boolean isAcceptable() {
        return false;
    }
}
