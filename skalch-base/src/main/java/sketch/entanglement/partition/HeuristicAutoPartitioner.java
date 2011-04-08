package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.HeuristicSearch;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class HeuristicAutoPartitioner extends TracePartitioner {

    class RankedSubset implements Comparable<RankedSubset> {
        final int score;
        final Set<Trace> traces;

        RankedSubset(Set<Trace> traces) {
            this.traces = traces;
            score =
                    HeuristicSearch.getEntanglementScore(new SATEntanglementAnalysis(
                            traces).getEntangledPartitions());
        }

        public int compareTo(RankedSubset o) {
            if (score < o.score) {
                return -1;
            } else if (score > o.score) {
                return 1;
            } else {
                return 0;
            }
        }

    }

    @Override
    public List<TraceSubset> getSubsets(TraceSubset p, String[] args) {
        if (args.length < 3) {
            List<TraceSubset> singleton = new ArrayList<TraceSubset>();
            singleton.add(p);
            return singleton;
        }
        int numSubsets = Integer.parseInt(args[0]);
        int branchingFactor = Integer.parseInt(args[1]);
        int minEntanglement = Integer.parseInt(args[2]);

        Set<Trace> traces = new HashSet<Trace>(p.getTraces());

        Set<Set<Trace>> subsets = getTopSubsets(traces, branchingFactor, minEntanglement);

        ArrayList<RankedSubset> rankedSubsets = new ArrayList<RankedSubset>();
        for (Set<Trace> subset : subsets) {
            rankedSubsets.add(new RankedSubset(subset));
        }
        Collections.sort(rankedSubsets);

        List<TraceSubset> returnedSubsets = new ArrayList<TraceSubset>();
        for (int i = 0; i < numSubsets && i < returnedSubsets.size(); i++) {
            returnedSubsets.add(new TraceSubset(new ArrayList<Trace>(
                    rankedSubsets.get(i).traces), "" + i, p));
        }
        return returnedSubsets;
    }

    private Set<Set<Trace>> getTopSubsets(Set<Trace> traces, int branchingFactor,
            int maxEntanglement)
    {

        Trace trace = traces.iterator().next();
        List<DynAngel> angelOrder = new ArrayList<DynAngel>();
        for (Event e : trace.getEvents()) {
            angelOrder.add(e.dynAngel);
        }

        SATEntanglementAnalysis satEA = new SATEntanglementAnalysis(traces);
        SimpleEntanglementAnalysis ea = new SimpleEntanglementAnalysis(traces);

        HeuristicSearch search = new HeuristicSearch(angelOrder, satEA, ea);
        search.partition();

        List<Set<Trace>> subsets = search.getTopSubsets(branchingFactor);
        Set<Set<Trace>> allGoodSubsets = new HashSet<Set<Trace>>();
        for (Set<Trace> subset : subsets) {
            if (!traces.equals(subset)) {
                SATEntanglementAnalysis subsetSatEA = new SATEntanglementAnalysis(subset);
                if (HeuristicSearch.getEntanglementScore(subsetSatEA.getEntangledPartitions()) < maxEntanglement)
                {
                    allGoodSubsets.add(subset);
                } else {
                    allGoodSubsets.addAll(getTopSubsets(subset, branchingFactor,
                            maxEntanglement));
                }
            }
        }
        return allGoodSubsets;
    }

    @Override
    public String toString() {
        return "Heuristic Auto Partitioner";
    }

}
