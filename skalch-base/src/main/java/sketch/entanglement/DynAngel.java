package sketch.entanglement;

public class DynAngel implements Comparable<DynAngel> {
    public final int staticAngelId;
    public final int execNum;

    public DynAngel(int staticAngelId, int execNum) {
        this.staticAngelId = staticAngelId;
        this.execNum = execNum;
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof DynAngel) {
            DynAngel otherLoc = (DynAngel) object;
            return staticAngelId == otherLoc.staticAngelId && execNum == otherLoc.execNum;
        }
        return false;
    }

    @Override
    public String toString() {
        return "" + staticAngelId + "[" + execNum + "]";
    }

    @Override
    public int hashCode() {
        return 1024 * staticAngelId + execNum;
    }

    public int compareTo(DynAngel otherAngel) {
        if (staticAngelId < otherAngel.staticAngelId) {
            return -1;
        } else if (staticAngelId > otherAngel.staticAngelId) {
            return 1;
        } else if (execNum < otherAngel.execNum) {
            return -1;
        } else if (execNum > otherAngel.execNum) {
            return 1;
        } else {
            return 0;
        }
    }
}
