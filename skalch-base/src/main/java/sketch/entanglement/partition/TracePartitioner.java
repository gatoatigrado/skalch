package sketch.entanglement.partition;

import java.util.List;

public abstract class TracePartitioner {

    public static TracePartitioner partitionTypes[] =
            { new SizePartitioner(), new StaticAngelPartitioner(),
                    new EntanglementPartitioner(), new HeuristicPartitioner(),
                    new HeuristicAutoPartitioner() };

    public abstract List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String args[]);

    @Override
    public abstract String toString();
}
