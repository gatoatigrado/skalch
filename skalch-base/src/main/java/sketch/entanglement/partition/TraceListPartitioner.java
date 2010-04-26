package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.List;

public abstract class TraceListPartitioner {

    public static List<TraceListPartitioner> getPartitionerTypes() {
        List<TraceListPartitioner> partitionTypes = new ArrayList<TraceListPartitioner>();
        partitionTypes.add(new FirstEntanglementPartitioner());
        partitionTypes.add(new SizePartitioner());
        return partitionTypes;
    }

    public abstract List<Partition> getTraceListPartition(Partition p);
}
