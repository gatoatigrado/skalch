package sketch.ui.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import sketch.ui.entanglement.Trace;

public class SizePartitioner implements TraceListPartitioner {

    List<Trace> traces;

    public SizePartitioner(List<Trace> traces) {
        this.traces = traces;
    }

    public List<List<Trace>> getTraceListPartition() {
        HashMap<Integer, List<Trace>> partitions =
                new HashMap<Integer, List<Trace>>();
        for (Trace trace : traces) {
            int size = traces.size();
            List<Trace> partition;
            if (partitions.containsKey(size)) {
                partition = partitions.get(size);
            } else {
                partition = new ArrayList<Trace>();
                partitions.put(size, partition);
            }
            partition.add(trace);
        }
        return new ArrayList<List<Trace>>(partitions.values());
    }
}
