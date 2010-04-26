package sketch.entanglement;

public class DynAngelPair implements Comparable<DynAngelPair> {
    final DynAngel loc1, loc2;

    public DynAngelPair(DynAngel loc1, DynAngel loc2) {
        if (loc1.compareTo(loc2) < 0) {
            this.loc1 = loc1;
            this.loc2 = loc2;
        } else {
            this.loc1 = loc2;
            this.loc2 = loc1;
        }
    }

    public DynAngel getFirst() {
        return loc1;
    }

    public DynAngel getSecond() {
        // TODO Auto-generated method stub
        return loc2;
    }

    @Override
    public int compareTo(DynAngelPair otherPair) {
        int firstComp = loc1.compareTo(otherPair.loc1);
        if (firstComp != 0) {
            return firstComp;
        } else {
            return loc2.compareTo(otherPair.loc2);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof DynAngelPair) {
            DynAngelPair d = (DynAngelPair) o;
            if (loc1.equals(d.loc1) && loc2.equals(d.loc2)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return loc1.hashCode() + loc2.hashCode();

    }

    @Override
    public String toString() {
        return "" + loc1 + " entangled with " + loc2;
    }

}
