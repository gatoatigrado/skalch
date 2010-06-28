package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import sketch.entanglement.Trace;

public class SizePartitioner extends TracePartitioner {

    @Override
    public List<Partition> getPartitions(Partition p, String args[]) {
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

        List<Partition> partitions = new ArrayList<Partition>();
        for (Integer size : sizePartitions.keySet()) {
            Partition partition =
                    new Partition(sizePartitions.get(size), size.toString(), p);
            partitions.add(partition);
        }
        return partitions;
    }

    @Override
    public String toString() {
        return "Size Partitioner";
    }
}
