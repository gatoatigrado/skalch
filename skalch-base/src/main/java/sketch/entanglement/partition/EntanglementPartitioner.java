package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import kodkod.util.ints.IntSet;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.entanglement.sat.TraceConverter;
import entanglement.MaxSupportFinder;
import entanglement.trace.Traces;

public class EntanglementPartitioner extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String[] args) {
        if (args.length == 0) {
            ArrayList<SubsetOfTraces> returnPartition = new ArrayList<SubsetOfTraces>();
            returnPartition.add(p);
            return returnPartition;
        }

        Set<Trace> traces = new HashSet<Trace>(p.getTraces());
        SATEntanglementAnalysis satEA = new SATEntanglementAnalysis(traces);

        TraceConverter converter = satEA.getTraceConverter();
        List<IntSet> oldSatPartitions = satEA.getEntangledIntSets();
        Set<Set<DynAngel>> oldPartitions =
                converter.getDynAngelPartitions(oldSatPartitions);

        System.out.println("Old partitions: " + oldSatPartitions);
        System.out.println("Old partitions: " + oldPartitions);

        List<List<DynAngel>> subpartitioning;
        if ("-s".equals(args[0])) {
            StringBuilder argString = new StringBuilder();
            for (int i = 1; i < args.length; i++) {
                argString.append(args[i]);
                argString.append(" ");
            }
            subpartitioning = DynAngel.parseDynAngelPartitioning(argString.toString());
            System.out.println("Partitions: " + subpartitioning);
        } else {
            StringBuilder argString = new StringBuilder();
            for (int i = 0; i < args.length; i++) {
                argString.append(args[i]);
                argString.append(" ");
            }
            DynAngel d = DynAngel.parseDynAngel(argString.toString());
            subpartitioning = new ArrayList<List<DynAngel>>();
            List<DynAngel> partition = new ArrayList<DynAngel>();
            partition.add(d);
            subpartitioning.add(partition);
        }

        Set<Set<DynAngel>> partitioning = new HashSet<Set<DynAngel>>();

        for (Set<DynAngel> partition : oldPartitions) {
            HashSet<DynAngel> partitionClone = new HashSet<DynAngel>(partition);
            for (List<DynAngel> subpartition : subpartitioning) {
                HashSet<DynAngel> projection = new HashSet<DynAngel>();
                for (DynAngel angel : subpartition) {
                    if (partition.contains(angel)) {
                        projection.add(angel);
                    }
                }
                if (!projection.isEmpty()) {
                    partitioning.add(projection);
                    partitionClone.removeAll(projection);
                }
            }
            if (!partitionClone.isEmpty()) {
                partitioning.add(partitionClone);
            }
        }

        List<IntSet> newPartitions = converter.getIntSetPartitions(partitioning);

        System.out.println("New partitions: " + partitioning);
        System.out.println("New partitions: " + newPartitions);

        int i = 0;
        Traces satTraces = converter.getTraces();
        List<SubsetOfTraces> subsets = new ArrayList<SubsetOfTraces>();

        long startTime = System.currentTimeMillis();

        for (Iterator<Traces> supports =
                MaxSupportFinder.findMaximalSupports(satTraces, oldSatPartitions,
                        newPartitions); supports.hasNext();)
        {
            Traces support = supports.next();
            System.out.println("Found support: " + support.size());
            List<Trace> subsetTraces = converter.convert(support);
            SubsetOfTraces subset = new SubsetOfTraces(subsetTraces, "" + i, p);
            subsets.add(subset);
            i++;
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Time to compute entanglement(ms): " + (endTime - startTime));

        return subsets;
    }

    @Override
    public String toString() {
        return "Entanglement Partitioner";
    }

}
