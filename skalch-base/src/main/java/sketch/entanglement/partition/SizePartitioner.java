package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import sketch.entanglement.Trace;

public class SizePartitioner extends TraceListPartitioner {

    public SizePartitioner() {}

    @Override
    public List<Partition> getTraceListPartition(Partition p) {
        List<Trace> traces = p.getTraces();
        HashMap<Integer, List<Trace>> sizePartitions =
                new HashMap<Integer, List<Trace>>();
        for (Trace trace : traces) {
            int size = traces.size();
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
}
