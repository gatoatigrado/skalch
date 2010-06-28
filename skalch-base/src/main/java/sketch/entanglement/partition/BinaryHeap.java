package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BinaryHeap<E extends Comparable<E>> {
    List<E> h = new ArrayList<E>();

    public BinaryHeap() {}

    // building heap in O(n)
    public BinaryHeap(E[] keys) {
        for (E key : keys) {
            h.add(key);
        }
        for (int pos = h.size() / 2 - 1; pos >= 0; pos--) {
            moveDown(pos);
        }
    }

    public List<E> getList() {
        return new ArrayList<E>(h);
    }

    public void add(E node) {
        h.add(node);
        moveUp(h.size() - 1);
    }

    private void moveUp(int pos) {
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (h.get(pos).compareTo(h.get(parent)) >= 0) {
                break;
            }
            Collections.swap(h, pos, parent);
            pos = parent;
        }
    }

    public E remove() {
        E removedNode = h.get(0);
        E lastNode = h.remove(h.size() - 1);
        if (!h.isEmpty()) {
            h.set(0, lastNode);
            moveDown(0);
        }
        return removedNode;
    }

    private void moveDown(int pos) {
        while (pos < h.size() / 2) {
            int child = 2 * pos + 1;
            if (child < h.size() - 1 && h.get(child).compareTo(h.get(child + 1)) > 0) {
                ++child;
            }
            if (h.get(pos).compareTo(h.get(child)) <= 0) {
                break;
            }
            Collections.swap(h, pos, child);
            pos = child;
        }
    }

    public int size() {
        return h.size();
    }
}
