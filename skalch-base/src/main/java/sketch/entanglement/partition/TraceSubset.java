package sketch.entanglement.partition;

import java.util.List;

import sketch.entanglement.Trace;

public class TraceSubset {

    private List<Trace> traces;
    private TraceSubset parent;
    private String name;

    public TraceSubset(List<Trace> traces, String name, TraceSubset parent) {
        this.traces = traces;
        this.name = name;
        this.parent = parent;
    }

    public List<Trace> getTraces() {
        return traces;
    }

    public String getPartitionName() {
        return name;
    }

    public boolean hasSameTraces(TraceSubset p) {
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
