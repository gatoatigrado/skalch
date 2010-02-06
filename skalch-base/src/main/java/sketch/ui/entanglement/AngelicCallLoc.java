package sketch.ui.entanglement;

public class AngelicCallLoc implements Comparable {
    public final int holeId;
    public final int numHoleExec;

    public AngelicCallLoc(int holeId, int numHoleExec) {
        this.holeId = holeId;
        this.numHoleExec = numHoleExec;
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof AngelicCallLoc) {
            AngelicCallLoc otherLoc = (AngelicCallLoc) object;
            return holeId == otherLoc.holeId
                    && numHoleExec == otherLoc.numHoleExec;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return 1024 * holeId + numHoleExec;
    }

    public int compareTo(Object o) {
        if (o instanceof AngelicCallLoc) {
            AngelicCallLoc otherLoc = (AngelicCallLoc) o;
            if (holeId < otherLoc.holeId) {
                return -1;
            } else if (holeId > otherLoc.holeId) {
                return 1;
            } else if (numHoleExec < otherLoc.numHoleExec) {
                return -1;
            } else if (numHoleExec > otherLoc.numHoleExec) {
                return 1;
            } else {
                return 0;
            }
        }
        throw new ClassCastException();
    }
}
