package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import sketch.entanglement.Trace;

public class SizePartitioner extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String args[]) {
        List<Trace> traces = p.getTraces();
        HashMap<Integer, List<Trace>> sizePartitions =
                new HashMap<Integer, List<Trace>>();
        for (Trace trace : traces) {
            int size = trace.size();
            List<Trace> partition;
            if (sizePartitions.containsKey(size)) {
                partition = sizePartitions.get(size);
            } else {
                partition = new ArrayList<Trace>();
                sizePartitions.put(size, partition);
            }
            partition.add(trace);
        }

        List<SubsetOfTraces> partitions = new ArrayList<SubsetOfTraces>();
        for (Integer size : sizePartitions.keySet()) {
            SubsetOfTraces partition =
                    new SubsetOfTraces(sizePartitions.get(size), size.toString(), p);
            partitions.add(partition);
        }
        return partitions;
    }

    @Override
    public String toString() {
        return "Size Partitioner";
    }
}
