package sketch.entanglement.sat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import kodkod.util.ints.IntIterator;
import kodkod.util.ints.IntSet;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.Trace;
import entanglement.EntanglementDetector;

public class SATEntanglementAnalysis {

    private static Map<AnalysisInput, AnalysisResult> cache =
            new HashMap<AnalysisInput, AnalysisResult>();

    private class AnalysisInput {

        public AnalysisInput(Set<Trace> traces, List<DynAngel> angels) {
            this.traces = traces;
            this.angels = angels;
        }

        final Set<Trace> traces;
        final List<DynAngel> angels;

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((angels == null) ? 0 : angels.hashCode());
            result = prime * result + ((traces == null) ? 0 : traces.hashCode());
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null) {
                return false;
            }
            if (getClass() != obj.getClass()) {
                return false;
            }
            AnalysisInput other = (AnalysisInput) obj;
            if (angels == null) {
                if (other.angels != null) {
                    return false;
                }
            } else if (!angels.equals(other.angels)) {
                return false;
            }
            if (traces == null) {
                if (other.traces != null) {
                    return false;
                }
            } else if (!traces.equals(other.traces)) {
                return false;
            }
            return true;
        }
    }

    private class AnalysisResult {
        final TraceConverter converter;
        final List<IntSet> entangledSets;

        public AnalysisResult(TraceConverter converter, List<IntSet> entangledSets) {
            this.converter = converter;
            this.entangledSets = entangledSets;
        }
    }

    final private Set<Trace> traces;
    final private List<DynAngel> angels;
    final private AnalysisInput ai;

    public SATEntanglementAnalysis(Set<Trace> traces) {
        this.traces = new HashSet<Trace>(traces);
        angels = getAngels(traces);
        ai = new AnalysisInput(this.traces, angels);
    }

    private List<DynAngel> getAngels(Set<Trace> traces) {
        Set<DynAngel> angels = new HashSet<DynAngel>();
        for (Trace trace : traces) {
            List<Event> events = trace.getEvents();
            for (Event event : events) {
                angels.add(event.dynAngel);
            }
        }
        List<DynAngel> angelList = new ArrayList<DynAngel>(angels);
        Collections.sort(angelList);
        return angelList;
    }

    public SATEntanglementAnalysis(Set<Trace> traces, List<DynAngel> angels) {
        this.traces = new HashSet<Trace>(traces);
        this.angels = new ArrayList<DynAngel>(angels);
        ai = new AnalysisInput(this.traces, this.angels);
    }

    public Set<Set<DynAngel>> getEntangledPartitions() {
        AnalysisResult result = cache.get(ai);
        if (result != null) {
            System.out.println("Found matching set of traces, no need to invoke sat based partitioner");
            return getPartitions(result.converter, result.entangledSets);
        }

        TraceConverter converter = new TraceConverter(angels, traces);
        System.out.println("Invoking sat based partitioner with " +
                converter.getTraces().size() + " traces.");

        List<IntSet> entangledSets =
                EntanglementDetector.entanglement(converter.getTraces());
        cache.put(ai, new AnalysisResult(converter, entangledSets));

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
        AnalysisResult result = cache.get(ai);
        if (result == null) {
            getEntangledPartitions();
            result = cache.get(ai);
        }
        return result.entangledSets;
    }

    public TraceConverter getTraceConverter() {
        AnalysisResult result = cache.get(ai);
        if (result == null) {
            getEntangledPartitions();
            result = cache.get(ai);
        }
        return result.converter;
    }
}
