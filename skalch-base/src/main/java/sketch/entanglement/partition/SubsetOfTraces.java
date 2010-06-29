package sketch.entanglement.partition;

import java.util.List;

import sketch.entanglement.Trace;

public class SubsetOfTraces {

    private List<Trace> traces;
    private SubsetOfTraces parent;
    private String name;

    public SubsetOfTraces(List<Trace> traces, String name, SubsetOfTraces parent) {
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

    public boolean hasSameTraces(SubsetOfTraces p) {
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
