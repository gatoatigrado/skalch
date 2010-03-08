package sketch.ui;

import javax.swing.JList;

/**
 * sorted UI list panel
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUiSortedList<T extends Comparable<T>> extends ScUiList<T> {
    public ScUiSortedList(JList list, Class<T[]> arrayCls, int maxEntries) {
        super(list, arrayCls, maxEntries);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void add(T element) {
        if (removeQueue.contains(element)) {
            removeQueue.remove(element);
        } else {
            int sz = listModel.getSize();
            int step = sz / 2;
            int insertIdx = sz / 2;
            // binary search
            while (insertIdx < sz) {
                step /= 2;
                T other = (T) listModel.getElementAt(insertIdx);
                boolean eltLtOther = (element.compareTo(other) < 0);
                if (eltLtOther) {
                    if (insertIdx == 0) {
                        // have element <= other[0], insert at position 0
                        break;
                    } else {
                        T otherLeft = (T) listModel.getElementAt(insertIdx - 1);
                        if (otherLeft.compareTo(element) <= 0) {
                            // otherLeft <= element < other
                            break;
                        } else {
                            // search left subtree to find closer values
                            insertIdx -= Math.max(1, step);
                        }
                    }
                } else {
                    if (insertIdx == sz - 1) {
                        insertIdx = sz;
                        break;
                    } else {
                        insertIdx += Math.max(1, step);
                    }
                }
            }
            if (listModel.getSize() >= maxEntries) {
                if (insertIdx < maxEntries - 1) {
                    listModel.add(insertIdx, element);
                    listModel.removeElementAt(listModel.getSize() - 1);
                }
            } else {
                listModel.add(insertIdx, element);
            }
        }
    }
}
