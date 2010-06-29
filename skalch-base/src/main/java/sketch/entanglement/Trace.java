package sketch.entanglement;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Trace {
    public final List<Event> events;

    public Trace() {
        events = new ArrayList<Event>();
    }

    public Trace(List<Event> e) {
        events = new ArrayList<Event>(e);
    }

    public void addEvent(int holeId, int numHoleExec, int numValues, int valueChosen) {
        events.add(new Event(holeId, numHoleExec, numValues, valueChosen));
    }

    public void addEvent(Event event) {
        events.add(event);
    }

    public Trace getSubTrace(Set<DynAngel> angelSubset) {
        Trace newTrace = new Trace();
        for (Event event : events) {
            if (angelSubset.contains(event.dynAngel)) {
                newTrace.addEvent(event);
            }
        }
        return newTrace;
    }

    @Override
    public String toString() {
        return events.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Trace) {
            Trace e = (Trace) o;
            if (e.events.size() == events.size()) {
                for (int i = 0; i < events.size(); i++) {
                    if (!events.get(i).equals(e.events.get(i))) {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        int sum = 0;
        for (Event event : events) {
            sum += event.hashCode();
        }
        return sum;
    }

    public List<Event> getEvents() {
        return new ArrayList<Event>(events);
    }

    public int size() {
        return events.size();
    }
}
