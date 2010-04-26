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

public class FirstEntanglementPartitioner extends TraceListPartitioner {

    public FirstEntanglementPartitioner() {}

    @Override
    public List<Partition> getTraceListPartition(Partition p) {
        List<Trace> traces = p.getTraces();

        EntanglementAnalysis ea = new EntanglementAnalysis(traces);
        Set<DynAngel> entangledDynAngels = ea.getEntangledAngels();
        if (entangledDynAngels.isEmpty()) {
            List<Partition> noPartition = new ArrayList<Partition>();
            noPartition.add(new Partition(traces, "noentang", p));
            return noPartition;
        }
        DynAngel firstEntangled = getFirstAngel(traces.get(0), entangledDynAngels);
        assert firstEntangled != null;
        System.out.println(firstEntangled);

        for (Trace trace : traces) {
            assert firstEntangled.equals(getFirstAngel(trace, entangledDynAngels));
        }
        HashSet<DynAngel> proj = new HashSet<DynAngel>();
        proj.add(firstEntangled);

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

        List<Partition> partitions = new ArrayList<Partition>();
        for (Trace projValue : projValuesToTraces.keySet()) {
            Partition partition =
                    new Partition(projValuesToTraces.get(projValue),
                            projValue.toString(), p);
            partitions.add(partition);
        }
        return partitions;
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
