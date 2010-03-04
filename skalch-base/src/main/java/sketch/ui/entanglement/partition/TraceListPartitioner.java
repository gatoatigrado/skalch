package sketch.ui.entanglement.partition;

import java.util.List;

import sketch.ui.entanglement.Trace;

public interface TraceListPartitioner {
    public List<List<Trace>> getTraceListPartition();
}
