package sketch.entanglement;

import java.util.ArrayList;
import java.util.List;

public class TwoSubsetIterator {

    private int subsetSize;
    private int totalSize;

    private ArrayList<Integer> curFirstSubset;
    private ArrayList<Integer> curSecondSubsetIndirect;
    private ArrayList<Integer> curSecondPossibleElmts;

    private ArrayList<Integer> nextFirstSubset;
    private ArrayList<Integer> nextSecondSubsetIndirect;
    private ArrayList<Integer> nextSecondPossibleElmts;
    private boolean firstRun;

    public TwoSubsetIterator(int subsetSize, int totalSize) {
        if (totalSize <= 1 || subsetSize <= 0) {
            nextFirstSubset = null;
            nextSecondSubsetIndirect = null;
            return;
        }

        if (subsetSize >= totalSize) {
            subsetSize = totalSize - 1;
        }

        this.subsetSize = subsetSize;
        this.totalSize = totalSize;

        nextFirstSubset = new ArrayList<Integer>();
        nextSecondPossibleElmts = new ArrayList<Integer>();
        nextSecondSubsetIndirect = new ArrayList<Integer>();

        nextFirstSubset.add(0);
        for (int i = 1; i < totalSize; i++) {
            nextSecondPossibleElmts.add(i);
        }
        nextSecondSubsetIndirect.add(0);

        firstRun = true;
    }

    public void next() {
        if (firstRun) {
            firstRun = false;
        }

        curFirstSubset = nextFirstSubset;
        curSecondSubsetIndirect = nextSecondSubsetIndirect;
        curSecondPossibleElmts = nextSecondPossibleElmts;

        updateNext();
    }

    public boolean hasNext() {
        return nextFirstSubset != null && nextSecondSubsetIndirect != null;
    }

    public List<Integer> firstSubset() {
        return new ArrayList<Integer>(curFirstSubset);
    }

    public List<Integer> secondSubset() {
        ArrayList<Integer> returnList = new ArrayList<Integer>();
        for (Integer index : curSecondSubsetIndirect) {
            returnList.add(curSecondPossibleElmts.get(index));
        }
        return returnList;
    }

    private void updateNext() {
        nextFirstSubset = new ArrayList<Integer>(curFirstSubset);
        nextSecondSubsetIndirect = new ArrayList<Integer>(curSecondSubsetIndirect);
        nextSecondPossibleElmts = new ArrayList<Integer>(curSecondPossibleElmts);

        // First try to increment one of the elements in the second set
        for (int i = nextSecondSubsetIndirect.size() - 1; i >= 0; i--) {
            // check if the current element can be incremented
            if (nextSecondSubsetIndirect.get(i) + 1 < nextSecondPossibleElmts.size()) {
                // check that incrementing the current element wont cause a
                // duplication
                if (i == nextSecondSubsetIndirect.size() - 1 ||
                        nextSecondSubsetIndirect.get(i) + 1 < nextSecondSubsetIndirect.get(i + 1))
                {
                    nextSecondSubsetIndirect.set(i, nextSecondSubsetIndirect.get(i) + 1);
                    for (int j = 1; j < nextSecondSubsetIndirect.size() - i; j++) {
                        nextSecondSubsetIndirect.set(i + j,
                                nextSecondSubsetIndirect.get(i) + j);
                    }
                    return;
                }
            }
        }

        // Try to increase the size of the second set
        if (nextSecondSubsetIndirect.size() < subsetSize &&
                nextSecondSubsetIndirect.size() < nextSecondPossibleElmts.size())
        {
            int newSize = nextSecondSubsetIndirect.size() + 1;
            resetSecondSet(newSize);
            return;
        }
        // Try to increment one of the elements in first set
        for (int i = nextFirstSubset.size() - 1; i >= 0; i--) {
            // check if the current element can be incremented
            if (nextFirstSubset.get(i) + 1 >= totalSize) {
                continue;
            }
            // check if incrementing the first element will make the size of second
            // subset zero
            if (i == 0 &&
                    totalSize - (nextFirstSubset.get(i) + 1 + nextFirstSubset.size()) <= 0)
            {
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
                updateSecondSet(1);
                return;
            }
        }

        // Try to increase the size of the first set
        if (nextFirstSubset.size() < subsetSize && nextFirstSubset.size() + 1 < totalSize)
        {
            int newSize = nextFirstSubset.size() + 1;
            nextFirstSubset.clear();
            for (int i = 0; i < newSize; i++) {
                nextFirstSubset.add(i);
            }
            updateSecondSet(1);
            return;
        }
        // End of lists
        nextFirstSubset = null;
        nextSecondSubsetIndirect = null;
        nextSecondPossibleElmts = null;
    }

    private void updateSecondSet(int secondSetSize) {
        nextSecondPossibleElmts.clear();

        for (int i = nextFirstSubset.get(0) + 1; i < totalSize; i++) {
            if (!nextFirstSubset.contains(i)) {
                nextSecondPossibleElmts.add(i);
            }
        }
        resetSecondSet(secondSetSize);
    }

    private void resetSecondSet(int secondSetSize) {
        nextSecondSubsetIndirect.clear();
        for (int i = 0; i < secondSetSize; i++) {
            nextSecondSubsetIndirect.add(i);
        }
    }
}
