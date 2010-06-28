package sketch.entanglement.partition;

import java.util.List;

public abstract class TracePartitioner {

    public static TracePartitioner partitionTypes[] =
            { new FirstEntanglementPartitioner(), new SizePartitioner(),
                    new StaticAngelPartitioner() };

    public abstract List<Partition> getPartitions(Partition p, String args[]);

    @Override
    public abstract String toString();
}
