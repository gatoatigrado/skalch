package sketch.entanglement;

public class Event {

    public final DynAngel dynAngel;
    public final int numValues;
    public final int valueChosen;

    public Event(int holeId, int numHoleExec, int numValues, int valueChosen) {
        dynAngel = new DynAngel(holeId, numHoleExec);
        this.numValues = numValues;
        this.valueChosen = valueChosen;
    }

    @Override
    public String toString() {
        return dynAngel.toString() + "=" + valueChosen;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Event) {
            Event a = (Event) o;
            if (dynAngel.equals(a.dynAngel) && valueChosen == a.valueChosen) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return dynAngel.hashCode() + valueChosen;
    }
}
