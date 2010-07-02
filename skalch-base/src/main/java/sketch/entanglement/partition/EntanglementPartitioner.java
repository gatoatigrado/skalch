package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import kodkod.util.ints.IntBitSet;
import kodkod.util.ints.IntIterator;
import kodkod.util.ints.IntSet;
import kodkod.util.ints.Ints;
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
        List<IntSet> oldPartitions = satEA.getEntangledIntSets();
        System.out.println("Old partitions: " + oldPartitions);
        List<IntSet> newPartitions;

        if ("-s".equals(args[0])) {
            StringBuilder argString = new StringBuilder();
            for (int i = 1; i < args.length; i++) {
                argString.append(args[i]);
                argString.append(" ");
            }
            List<List<DynAngel>> partitioning =
                    DynAngel.parseDynAngelPartitioning(argString.toString());
            System.out.println("Partitions: " + partitioning);
            newPartitions = new ArrayList<IntSet>();

            for (List<DynAngel> partition : partitioning) {
                if (partition.size() == 1) {
                    newPartitions.add(Ints.singleton(converter.getIndex(partition.get(0))));
                } else {
                    int maxValue = -1;
                    List<Integer> indexes = new ArrayList<Integer>();
                    for (DynAngel angel : partition) {
                        int index = converter.getIndex(angel);
                        indexes.add(index);
                        if (index > maxValue) {
                            maxValue = index;
                        }
                    }
                    IntSet partitionIndexes = new IntBitSet(maxValue + 1);
                    for (Integer index : indexes) {
                        partitionIndexes.add(index);
                    }
                    newPartitions.add(partitionIndexes);
                }
            }
        } else {
            StringBuilder argString = new StringBuilder();
            for (int i = 0; i < args.length; i++) {
                argString.append(args[i]);
                argString.append(" ");
            }
            DynAngel d = DynAngel.parseDynAngel(argString.toString());

            int index = converter.getIndex(d);
            newPartitions = new ArrayList<IntSet>();

            for (int i = 0; i < oldPartitions.size(); i++) {
                IntSet curPartition = oldPartitions.get(i);
                if (curPartition.contains(index) && curPartition.size() > 1) {
                    IntSet singleton = Ints.singleton(index);
                    IntSet rest = new IntBitSet(curPartition.max() + 1);
                    for (IntIterator it = curPartition.iterator(); it.hasNext();) {
                        int element = it.next();
                        if (element != index) {
                            rest.add(element);
                        }
                    }
                    newPartitions.add(singleton);
                    newPartitions.add(rest);
                } else {
                    newPartitions.add(curPartition);
                }
            }
        }
        System.out.println("New partitions: " + newPartitions);

        int i = 0;
        Traces satTraces = converter.getTraces();
        List<SubsetOfTraces> subsets = new ArrayList<SubsetOfTraces>();

        for (Iterator<Traces> supports =
                MaxSupportFinder.findMaximalSupports(satTraces, oldPartitions,
                        newPartitions); supports.hasNext();)
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
