package sketch.entanglement.partition;

import java.util.List;

public class TracePartitionerWithArgs extends TracePartitioner {

    private TracePartitioner partitioner;
    private String[] storedArgs;

    public TracePartitionerWithArgs(TracePartitioner partitioner, String storedArgs[]) {
        this.partitioner = partitioner;
        this.storedArgs = storedArgs;
    }

    @Override
    public List<Partition> getPartitions(Partition p, String[] args) {
        return partitioner.getPartitions(p, storedArgs);
    }

    @Override
    public String toString() {
        return partitioner.toString() + "(with args)";
    }
}
