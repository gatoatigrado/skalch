package sketch.ui;

import java.lang.reflect.Array;
import java.util.HashSet;
import java.util.List;

import javax.swing.DefaultListModel;
import javax.swing.JList;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

/**
 * generic wrapper for a list view. can remove elements before they are added.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUiList<T> {
    private static final long serialVersionUID = -21488839374096350L;
    protected DefaultListModel listModel = new DefaultListModel();
    protected JList list;
    protected Class<T[]> arrayCls;
    protected int maxEntries;
    protected HashSet<T> removeQueue = new HashSet<T>();

    public ScUiList(JList list, Class<T[]> arrayCls, int maxEntries) {
        this.list = list;
        list.setModel(listModel);
        this.arrayCls = arrayCls;
        this.maxEntries = maxEntries;
    }

    public void add(T element) {
        if (removeQueue.contains(element)) {
            removeQueue.remove(element);
        } else {
            if (listModel.getSize() < maxEntries) {
                listModel.addElement(element);
            }
        }
    }

    public void addAll(List<? extends T> elements) {
        ListDataListener[] dataListeners = listModel.getListDataListeners();
        for (ListDataListener l : dataListeners) {
            listModel.removeListDataListener(l);
        }

        int curSize = listModel.size();

        for (T element : elements) {
            add(element);
        }

        for (ListDataListener l : dataListeners) {
            listModel.addListDataListener(l);
            if (curSize < listModel.size()) {
                l.intervalAdded(new ListDataEvent(listModel,
                        ListDataEvent.INTERVAL_ADDED, curSize, listModel.size() - 1));
            }
        }
    }

    @SuppressWarnings("unchecked")
    public T[] getSelected() {
        if (arrayCls == null) {
            return (T[]) new Object[0];
        }
        Object[] asObj = list.getSelectedValues();
        return arrayCopy(asObj, asObj.length, arrayCls);
    }

    public Object[] getAll() {
        return listModel.toArray();
    }

    public void remove(T elt) {
        if (!listModel.removeElement(elt)) {
            removeQueue.add(elt);
        }
    }

    public void removeAll() {
        listModel.removeAllElements();
    }

    @SuppressWarnings("unchecked")
    public T selectNext(T elt) {
        int idx = Math.min(listModel.indexOf(elt) + 1, listModel.size() - 1);
        list.setSelectedIndex(idx);
        return (T) listModel.elementAt(idx);
    }

    @SuppressWarnings("unchecked")
    public T selectPrev(T elt) {
        int idx = Math.max(0, listModel.indexOf(elt) - 1);
        list.setSelectedIndex(idx);
        return (T) listModel.elementAt(idx);
    }

    public void setSelected(T elt) {
        list.setSelectedIndex(listModel.indexOf(elt));
    }

    // 1.5 compatibility
    @SuppressWarnings("unchecked")
    public static <NEW, OLD> NEW[] arrayCopy(OLD[] original, int newLength,
            Class<? extends NEW[]> newType)
    {
        NEW[] copy = (NEW[]) Array.newInstance(newType.getComponentType(), newLength);
        System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
        return copy;
    }
}
