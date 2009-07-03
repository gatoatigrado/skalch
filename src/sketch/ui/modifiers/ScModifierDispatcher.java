package sketch.ui.modifiers;

import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

/**
 * A class which has bound variables. This is useful for the synthesis list,
 * where different synthesizers may have significantly different functionalities
 * for showing in-progress synthesis.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScModifierDispatcher implements
        Comparable<ScModifierDispatcher>
{
    public ScUiThread ui_thread;
    public ScUiList<ScModifierDispatcher> list;

    public ScModifierDispatcher(ScUiThread ui_thread,
            ScUiList<ScModifierDispatcher> list)
    {
        this.ui_thread = ui_thread;
        this.list = list;
    }

    public void add() {
        list.add(this);
    }

    public void dispatch() {
        try {
            enqueue(new ScUiModifier(ui_thread, get_modifier()));
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
            list.remove(this);
        }
    }

    public int getCost() {
        return 1000000000;
    }

    public int compareTo(ScModifierDispatcher other) {
        int my_cost = getCost();
        int other_cost = other.getCost();
        if (my_cost < other_cost) {
            return -1;
        } else if (my_cost == other_cost) {
            return 0;
        } else {
            return -1;
        }
    }

    public abstract ScUiModifierInner get_modifier();

    @Override
    public abstract String toString();

    public abstract void enqueue(ScUiModifier m) throws ScUiQueueableInactive;

    public boolean isAcceptable() {
        return false;
    }
}
