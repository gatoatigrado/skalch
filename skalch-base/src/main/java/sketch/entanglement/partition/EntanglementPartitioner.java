package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import kodkod.util.ints.IntSet;
import kodkod.util.ints.Ints;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.TraceConverter;
import entanglement.EntanglementDetector;
import entanglement.MaxSupportFinder;
import entanglement.trace.Traces;

public class EntanglementPartitioner extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String[] args) {
        if (args.length < 2) {
            ArrayList<SubsetOfTraces> returnPartition = new ArrayList<SubsetOfTraces>();
            return returnPartition;
        }

        int staticId = Integer.parseInt(args[0]);
        int execNum = Integer.parseInt(args[1]);
        DynAngel d = new DynAngel(staticId, execNum);

        Set<Trace> traces = new HashSet<Trace>(p.getTraces());
        TraceConverter converter = new TraceConverter(traces);
        Traces satTraces = converter.getTraces();

        int index = converter.getIndex(d);

        List<IntSet> angelPartitions = EntanglementDetector.entanglement(satTraces);
        System.out.println("Old partitions: " + angelPartitions);

        for (int i = 0; i < angelPartitions.size(); i++) {
            IntSet curPartition = angelPartitions.get(i);
            if (curPartition.contains(index) && curPartition.size() > 1) {
                curPartition.remove(index);
                IntSet singleton = Ints.singleton(index);
                angelPartitions.add(singleton);
                break;
            }
        }
        System.out.println("New partitions: " + angelPartitions);

        int i = 0;
        List<SubsetOfTraces> subsets = new ArrayList<SubsetOfTraces>();
        for (Iterator<Traces> supports =
                MaxSupportFinder.findMaximalSupports(satTraces, angelPartitions); supports.hasNext();)
        {
            Traces support = supports.next();
            System.out.println("Found support: " + support.size());
            List<Trace> subsetTraces = converter.convert(support);
            SubsetOfTraces subset = new SubsetOfTraces(subsetTraces, "" + i, p);
            subsets.add(subset);
            i++;
        }
        return subsets;
    }

    @Override
    public String toString() {
        return "Entanglement Partitioner";
    }

}
