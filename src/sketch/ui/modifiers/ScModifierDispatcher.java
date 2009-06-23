package sketch.ui.modifiers;

import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;
import sketch.util.DebugOut;

/**
 * A class which has bound variables. This is useful for the synthesis list,
 * where different synthesizers may have significantly different functionalities
 * for showing in-progress synthesis.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScModifierDispatcher {
    public ScUiThread ui_thread;
    public ScUiList<ScModifierDispatcher> list;

    public ScModifierDispatcher(ScUiThread ui_thread,
            ScUiList<ScModifierDispatcher> list)
    {
        this.ui_thread = ui_thread;
        this.list = list;
    }

    public void add() {
        try {
            final ScModifierDispatcher elt_to_add = this;
            new ScUiModifier(ui_thread, new ScUiModifierInner() {
                @Override
                public void apply() {
                    list.add(elt_to_add);
                }
            }).enqueueTo();
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
            DebugOut.assertFalse();
        }
    }

    public void dispatch() {
        try {
            enqueue(new ScUiModifier(ui_thread, get_modifier()));
        } catch (ScUiQueueableInactive e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            list.remove(this);
        }
    }

    public abstract ScUiModifierInner get_modifier();

    @Override
    public abstract String toString();

    public abstract void enqueue(ScUiModifier m) throws ScUiQueueableInactive;
}
