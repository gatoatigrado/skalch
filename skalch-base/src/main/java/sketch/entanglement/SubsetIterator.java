package sketch.entanglement;

import java.util.ArrayList;
import java.util.List;

public class SubsetIterator {

    private int totalSize;
    private int minSubsetSize;
    private int maxSubsetSize;

    private ArrayList<Integer> curFirstSubset;
    private ArrayList<Integer> nextFirstSubset;

    public SubsetIterator(int minSubsetSize, int maxSubsetSize, int totalSize) {
        if (totalSize <= 1 || maxSubsetSize <= 0 || minSubsetSize > maxSubsetSize) {
            nextFirstSubset = null;
            return;
        }

        if (maxSubsetSize >= totalSize) {
            maxSubsetSize = totalSize;
        }

        if (minSubsetSize <= 0) {
            minSubsetSize = 1;
        }

        if (maxSubsetSize < minSubsetSize) {
            maxSubsetSize = minSubsetSize;
        }

        this.minSubsetSize = minSubsetSize;
        this.maxSubsetSize = maxSubsetSize;
        this.totalSize = totalSize;

        nextFirstSubset = new ArrayList<Integer>();
        for (int i = 0; i < this.minSubsetSize; i++) {
            nextFirstSubset.add(i);
        }
    }

    public List<Integer> next() {
        curFirstSubset = nextFirstSubset;
        updateNext();
        return new ArrayList<Integer>(curFirstSubset);
    }

    public boolean hasNext() {
        return nextFirstSubset != null;
    }

    private void updateNext() {
        nextFirstSubset = new ArrayList<Integer>(curFirstSubset);

        // Try to increment one of the elements in first set
        for (int i = nextFirstSubset.size() - 1; i >= 0; i--) {
            // check if the current element can be incremented
            if (nextFirstSubset.get(i) + 1 >= totalSize) {
                continue;
            }

            // check that incrementing the current element wont cause a duplication
            if (i == nextFirstSubset.size() - 1 ||
                    nextFirstSubset.get(i) + 1 < nextFirstSubset.get(i + 1))
            {
                nextFirstSubset.set(i, nextFirstSubset.get(i) + 1);
                for (int j = 1; j < nextFirstSubset.size() - i; j++) {
                    nextFirstSubset.set(i + j, nextFirstSubset.get(i) + j);
                }
                return;
            }
        }

        // Try to increase the size of the first set
        if (nextFirstSubset.size() < maxSubsetSize) {
            int newSize = nextFirstSubset.size() + 1;
            nextFirstSubset.clear();
            for (int i = 0; i < newSize; i++) {
                nextFirstSubset.add(i);
            }
            return;
        }
        // End of lists
        nextFirstSubset = null;
    }
}
