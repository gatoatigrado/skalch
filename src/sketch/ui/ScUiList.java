package sketch.ui;

import java.lang.reflect.Array;
import java.util.HashSet;

import javax.swing.DefaultListModel;
import javax.swing.JList;

/**
 * generic wrapper for a list view. can remove elements before they are added.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiList<T> {
    private static final long serialVersionUID = -21488839374096350L;
    protected DefaultListModel list_model = new DefaultListModel();
    protected JList list;
    protected Class<T[]> array_cls;
    protected HashSet<T> remove_queue = new HashSet<T>();

    public ScUiList(JList list, Class<T[]> array_cls) {
        this.list = list;
        list.setModel(list_model);
        this.array_cls = array_cls;
    }

    public void add(T element) {
        if (remove_queue.contains(element)) {
            remove_queue.remove(element);
        } else {
            list_model.addElement(element);
        }
    }

    @SuppressWarnings("unchecked")
    public T[] getSelected() {
        if (array_cls == null) {
            return (T[]) new Object[0];
        }
        Object[] as_obj = list.getSelectedValues();
        return arrayCopy(as_obj, as_obj.length, array_cls);
    }

    public void remove(T elt) {
        if (!list_model.removeElement(elt)) {
            remove_queue.add(elt);
        }
    }

    // 1.5 compatibility
    @SuppressWarnings("unchecked")
    public static <NEW, OLD> NEW[] arrayCopy(OLD[] original, int newLength,
            Class<? extends NEW[]> newType)
    {
        NEW[] copy =
                (NEW[]) Array
                        .newInstance(newType.getComponentType(), newLength);
        System.arraycopy(original, 0, copy, 0, Math.min(original.length,
                newLength));
        return copy;
    }
}
