package sketch.entanglement.partition;

import java.util.List;

import sketch.entanglement.Trace;

public interface TraceListPartitioner {
    public List<List<Trace>> getTraceListPartition();
}
