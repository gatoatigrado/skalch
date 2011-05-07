package sketch.entanglement.partition;

import java.util.List;

import sketch.entanglement.deprecated.HeuristicAutoPartitioner;
import sketch.entanglement.deprecated.HeuristicPartitioner;

public abstract class TracePartitioner {

    public static TracePartitioner partitionTypes[] =
            { new SizePartitioner(), new StaticAngelPartitioner(),
                    new EntanglementPartitioner(), new HeuristicPartitioner(),
                    new HeuristicAutoPartitioner() };

    public abstract List<TraceSubset> getSubsets(TraceSubset p, String args[]);

    @Override
    public abstract String toString();
}
