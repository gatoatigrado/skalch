package sketch.ui;

import javax.swing.JList;

/**
 * sorted UI list panel
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiSortedList<T extends Comparable<T>> extends ScUiList<T> {
    public ScUiSortedList(JList list, Class<T[]> array_cls, int max_entries) {
        super(list, array_cls, max_entries);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void add(T element) {
        if (remove_queue.contains(element)) {
            remove_queue.remove(element);
        } else {
            int sz = list_model.getSize();
            int step = sz / 2;
            int insert_idx = sz / 2;
            // binary search
            while (insert_idx < sz) {
                step /= 2;
                T other = (T) list_model.getElementAt(insert_idx);
                boolean elt_leq_other = (element.compareTo(other) <= 0);
                if (elt_leq_other) {
                    if (insert_idx == 0) {
                        // have element <= other[0], insert at position 0
                        break;
                    } else {
                        T other_left =
                                (T) list_model.getElementAt(insert_idx - 1);
                        if (other_left.compareTo(element) <= 0) {
                            // other_left <= element <= other
                            break;
                        } else {
                            // search left subtree to find closer values
                            insert_idx -= Math.max(1, step);
                        }
                    }
                } else {
                    if (insert_idx == sz - 1) {
                        insert_idx = sz;
                        break;
                    } else {
                        insert_idx += Math.max(1, step);
                    }
                }
            }
            list_model.add(insert_idx, element);
            if (list_model.getSize() >= max_entries) {
                list_model.removeElementAt(list_model.getSize() - 1);
            }
        }
    }
}
