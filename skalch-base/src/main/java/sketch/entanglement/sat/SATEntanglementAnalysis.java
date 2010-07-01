package sketch.entanglement.sat;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import kodkod.util.ints.IntIterator;
import kodkod.util.ints.IntSet;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Trace;
import entanglement.EntanglementDetector;

public class SATEntanglementAnalysis {

    private static Map<Set<Trace>, AnalysisResult> cache =
            new HashMap<Set<Trace>, AnalysisResult>();

    private class AnalysisResult {

        final TraceConverter converter;
        final List<IntSet> entangledSets;

        public AnalysisResult(TraceConverter converter, List<IntSet> entangledSets) {
            this.converter = converter;
            this.entangledSets = entangledSets;
        }
    }

    final private Set<Trace> traces;

    public SATEntanglementAnalysis(Set<Trace> traces) {
        this.traces = new HashSet<Trace>(traces);
    }

    public Set<Set<DynAngel>> getEntangledPartitions() {
        AnalysisResult result = cache.get(traces);
        if (result != null) {
            System.out.println("Found matching set of traces, no need to invoke sat based partitioner");
            return getPartitions(result.converter, result.entangledSets);
        }

        TraceConverter converter = new TraceConverter(traces);
        System.out.println("Invoking sat based partitioner with " +
                converter.getTraces().size() + " traces.");

        List<IntSet> entangledSets =
                EntanglementDetector.entanglement(converter.getTraces());
        cache.put(traces, new AnalysisResult(converter, entangledSets));

        Set<Set<DynAngel>> subsets = getPartitions(converter, entangledSets);
        return subsets;
    }

    private Set<Set<DynAngel>> getPartitions(TraceConverter converter,
            List<IntSet> entangledSets)
    {
        Set<Set<DynAngel>> subsets = new HashSet<Set<DynAngel>>();
        for (IntSet entangledSet : entangledSets) {
            Set<DynAngel> subset = new HashSet<DynAngel>();
            for (IntIterator iterator = entangledSet.iterator(); iterator.hasNext();) {
                int index = iterator.next();
                subset.add(converter.getAngel(index));
            }
            subsets.add(subset);
        }
        return subsets;
    }

    public List<IntSet> getEntangledIntSets() {
        AnalysisResult result = cache.get(traces);
        if (result == null) {
            getEntangledPartitions();
            result = cache.get(traces);
        }
        return result.entangledSets;
    }

    public TraceConverter getTraceConverter() {
        AnalysisResult result = cache.get(traces);
        if (result == null) {
            getEntangledPartitions();
            result = cache.get(traces);
        }
        return result.converter;
    }
}
