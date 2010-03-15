package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.EntanglementAnalysis;
import sketch.entanglement.Event;
import sketch.entanglement.Trace;

public class FirstEntanglementPartitioner implements TraceListPartitioner {

    List<Trace> traces;

    public FirstEntanglementPartitioner(List<Trace> traces) {
        this.traces = traces;
    }

    public List<List<Trace>> getTraceListPartition() {
        EntanglementAnalysis ea = new EntanglementAnalysis(traces);
        Set<DynAngel> entangledDynAngels = ea.getEntangledAngels();
        if (entangledDynAngels.isEmpty()) {
            List<List<Trace>> noPartition = new ArrayList<List<Trace>>();
            noPartition.add(traces);
            return noPartition;
        }

        DynAngel firstEntangled = getFirstAngel(traces.get(0), entangledDynAngels);
        assert firstEntangled != null;

        for (Trace trace : traces) {
            assert firstEntangled.equals(getFirstAngel(trace, entangledDynAngels));
        }
        HashSet<DynAngel> proj = new HashSet<DynAngel>();
        Set<Trace> projValues = ea.getPossibleValues(proj);
        HashMap<Trace, List<Trace>> projValuesToTraces =
                new HashMap<Trace, List<Trace>>();
        for (Trace projValue : projValues) {
            projValuesToTraces.put(projValue, new ArrayList<Trace>());
        }

        for (Trace trace : traces) {
            Trace subtrace = trace.getSubTrace(proj);
            List<Trace> partition = projValuesToTraces.get(subtrace);
            partition.add(trace);
        }
        return new ArrayList<List<Trace>>(projValuesToTraces.values());
    }

    private DynAngel getFirstAngel(Trace trace, Set<DynAngel> entangledDynAngels) {
        for (Event event : trace.getEvents()) {
            if (entangledDynAngels.contains(event.dynAngel)) {
                return event.dynAngel;
            }
        }
        return null;
    }
}
