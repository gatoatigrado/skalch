package sketch.entanglement.deprecated;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.TracePartitioner;
import sketch.entanglement.partition.TraceSubset;
import sketch.entanglement.sat.SATEntanglementAnalysis;

@Deprecated
public class HeuristicPartitioner extends TracePartitioner {

    @Override
    public List<TraceSubset> getSubsets(TraceSubset p, String[] args) {
        if (args.length == 0) {
            List<TraceSubset> singleton = new ArrayList<TraceSubset>();
            singleton.add(p);
            return singleton;
        }
        int numSubsets = Integer.parseInt(args[0]);

        Set<Trace> traces = new HashSet<Trace>(p.getTraces());

        Trace trace = traces.iterator().next();
        List<DynAngel> angelOrder = new ArrayList<DynAngel>();
        for (Event e : trace.getEvents()) {
            angelOrder.add(e.dynAngel);
        }

        SATEntanglementAnalysis satEA = new SATEntanglementAnalysis(traces);
        SimpleEntanglementAnalysis ea = new SimpleEntanglementAnalysis(traces);

        HeuristicSearch search = new HeuristicSearch(angelOrder, satEA, ea);
        search.partition();

        List<Set<Trace>> topSubsets = search.getTopSubsets(numSubsets);
        List<TraceSubset> returnedSubsets = new ArrayList<TraceSubset>();
        for (int i = 0; i < numSubsets && i < topSubsets.size(); i++) {
            returnedSubsets.add(new TraceSubset(
                    new ArrayList<Trace>(topSubsets.get(i)), "" + i, p));
        }
        return returnedSubsets;
    }

    @Override
    public String toString() {
        return "Heuristic Partitioner";
    }

}
