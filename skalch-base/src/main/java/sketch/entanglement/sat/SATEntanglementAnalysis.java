package sketch.entanglement.sat;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import kodkod.util.ints.IntIterator;
import kodkod.util.ints.IntSet;
import sketch.entanglement.DynAngel;
import sketch.entanglement.EntangledPartitions;
import sketch.entanglement.Trace;
import entanglement.EntanglementDetector;

public class SATEntanglementAnalysis {

    final private TraceConverter converter;

    public SATEntanglementAnalysis(Set<Trace> traces) {
        converter = new TraceConverter(traces);

        System.out.println("Created sat based partitioner with:");
        System.out.println("Simple traces: " + converter.getTraces().size());
    }

    public EntangledPartitions getEntangledPartitions(int n) {
        String p = System.getProperty("java.library.path");
        System.out.println(p);
        List<IntSet> entangledSets =
                EntanglementDetector.entanglement(converter.getTraces());
        Set<Set<DynAngel>> subsets = new HashSet<Set<DynAngel>>();
        for (IntSet entangledSet : entangledSets) {
            Set<DynAngel> subset = new HashSet<DynAngel>();
            for (IntIterator iterator = entangledSet.iterator(); iterator.hasNext();) {
                int index = iterator.next();
                subset.add(converter.getAngel(index));
            }
            subsets.add(subset);
        }
        return new EntangledPartitions(subsets, new HashSet<Set<DynAngel>>());
    }
}
