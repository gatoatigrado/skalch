package sketch.entanglement.partition;

import java.util.List;

import sketch.entanglement.Trace;

public class Partition {

    private List<Trace> traces;
    private Partition parent;
    private String name;

    public Partition(List<Trace> traces, String name, Partition parent) {
        this.traces = traces;
        this.name = name;
        this.parent = parent;
    }

    public List<Trace> getTraces() {
        return traces;
    }

    public String getParitionName() {
        return name;
    }

    public boolean hasSameTraces(Partition p) {
        return traces.containsAll(p.getTraces());
    }

    @Override
    public String toString() {
        String returnString = "";
        if (parent != null) {
            returnString += parent.toString() + ",";
        }
        returnString += name;
        return returnString;
    }
}
