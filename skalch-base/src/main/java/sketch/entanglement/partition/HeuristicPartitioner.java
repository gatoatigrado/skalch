package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.HeuristicSearch;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class HeuristicPartitioner extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String[] args) {
        if (args.length == 0) {
            List<SubsetOfTraces> singleton = new ArrayList<SubsetOfTraces>();
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
        List<SubsetOfTraces> returnedSubsets = new ArrayList<SubsetOfTraces>();
        for (int i = 0; i < numSubsets && i < topSubsets.size(); i++) {
            returnedSubsets.add(new SubsetOfTraces(
                    new ArrayList<Trace>(topSubsets.get(i)), "" + i, p));
        }
        return returnedSubsets;
    }

    @Override
    public String toString() {
        return "Heuristic Partitioner";
    }

}
