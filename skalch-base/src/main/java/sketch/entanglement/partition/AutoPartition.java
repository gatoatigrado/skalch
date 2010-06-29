package sketch.entanglement.partition;

import java.sql.Time;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.EntangledPartitions;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class AutoPartition extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String[] args) {
        if (args.length < 3) {
            ArrayList<SubsetOfTraces> singletonList = new ArrayList<SubsetOfTraces>();
            singletonList.add(p);
            return singletonList;
        }
        int depth = Integer.parseInt(args[0]);
        int branching = Integer.parseInt(args[1]);
        int subsetSize = Integer.parseInt(args[2]);

        BinaryHeap<PartitionInfo> curSolutionHeap = new BinaryHeap<PartitionInfo>();
        BinaryHeap<PartitionInfo> allSolutionHeap = new BinaryHeap<PartitionInfo>();

        HashMap<Integer, List<SubsetOfTraces>> seenPartitions =
                new HashMap<Integer, List<SubsetOfTraces>>();

        List<SubsetOfTraces> newSolutionList = new ArrayList<SubsetOfTraces>();
        newSolutionList.add(p);
        addToSeenPartitions(seenPartitions, p);

        for (int i = 0; i < depth; i++) {
            System.out.println("Depth " + i);

            for (SubsetOfTraces newSolution : newSolutionList) {
                curSolutionHeap.add(new PartitionInfo(newSolution, subsetSize));
            }
            newSolutionList.clear();

            int numSolutionsProcessed = 0;
            // go through each current solution
            while (curSolutionHeap.size() > 0 && numSolutionsProcessed < branching) {
                numSolutionsProcessed++;
                PartitionInfo solution = curSolutionHeap.remove();
                System.out.println("Partitioning " +
                        solution.partition.getPartitionName());
                allSolutionHeap.add(solution);

                List<TracePartitioner> partitioners = getAllTracePartitioners(solution);
                // go through partitioners suitable for this partition
                for (TracePartitioner partitioner : partitioners) {
                    List<SubsetOfTraces> partitions =
                            partitioner.getSubsets(solution.partition, null);
                    // add each new partition to solutions list
                    for (SubsetOfTraces partition : partitions) {
                        if (!hasBeenSeenPartition(seenPartitions, partition)) {
                            System.out.println("New partition " + partition);
                            addToSeenPartitions(seenPartitions, partition);
                            newSolutionList.add(partition);
                        }
                    }
                }
            }

            // empty out the current solution heap
            while (curSolutionHeap.size() > 0) {
                allSolutionHeap.add(curSolutionHeap.remove());
            }

        }

        ArrayList<SubsetOfTraces> returnList = new ArrayList<SubsetOfTraces>();
        while (curSolutionHeap.size() > 0) {
            returnList.add(curSolutionHeap.remove().partition);
        }
        returnList.addAll(newSolutionList);
        return returnList;
    }

    private boolean hasBeenSeenPartition(
            HashMap<Integer, List<SubsetOfTraces>> seenPartitions, SubsetOfTraces partition)
    {
        int size = partition.getTraces().size();
        if (seenPartitions.containsKey(size)) {
            for (SubsetOfTraces sameSizedPartition : seenPartitions.get(size)) {
                if (partition.hasSameTraces(sameSizedPartition)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void addToSeenPartitions(HashMap<Integer, List<SubsetOfTraces>> seenPartitions,
            SubsetOfTraces partition)
    {
        int size = partition.getTraces().size();
        if (seenPartitions.containsKey(size)) {
            seenPartitions.get(size).add(partition);
        } else {
            List<SubsetOfTraces> sameSizePartitions = new ArrayList<SubsetOfTraces>();
            sameSizePartitions.add(partition);
            seenPartitions.put(size, sameSizePartitions);
        }
    }

    @Override
    public String toString() {
        return "AutoPartition";
    }

    private List<TracePartitioner> getAllTracePartitioners(PartitionInfo info) {
        List<Trace> traces = info.partition.getTraces();
        EntangledPartitions es = info.entanglementSubsets;
        SATEntanglementAnalysis ea = info.entanglementAnalysis;

        List<TracePartitioner> partitioners = new ArrayList<TracePartitioner>();

        // Add size partitioner if appropriate
        boolean diffSizeTraces = false;
        if (!traces.isEmpty()) {
            int size = traces.get(0).size();
            for (Trace t : traces) {
                if (size != t.size()) {
                    diffSizeTraces = true;
                    break;
                }
            }
        }

        if (diffSizeTraces) {
            partitioners.add(new SizePartitioner());
        }

        // Add static angel partitioners
        List<DynAngel> entangledAngels = new ArrayList<DynAngel>();
        for (Set<DynAngel> entangledSet : es.entangledSubsets) {
            entangledAngels.addAll(entangledSet);
        }
        for (Set<DynAngel> unentangledSet : es.unentangledSubsets) {
            if (unentangledSet.size() > 1) {
                entangledAngels.addAll(unentangledSet);
            }
        }

        HashSet<Integer> staticAngels = new HashSet<Integer>();
        for (DynAngel angel : entangledAngels) {
            staticAngels.add(angel.staticAngelId);
        }

        for (Integer staticAngel : staticAngels) {
            String args[] = { staticAngel.toString() };
            partitioners.add(new TracePartitionerWithArgs(new StaticAngelPartitioner(),
                    args));
        }

        return partitioners;
    }
}

class PartitionInfo implements Comparable<PartitionInfo> {
    public SATEntanglementAnalysis entanglementAnalysis;
    public EntangledPartitions entanglementSubsets;
    public final SubsetOfTraces partition;
    public final int entanglementScore;

    public PartitionInfo(SubsetOfTraces p, int subsetSize) {
        System.out.println("Calculating entanglement score for partition " +
                p.getPartitionName() + " (" + new Time(System.currentTimeMillis()) + ")");

        partition = p;
        entanglementAnalysis =
                new SATEntanglementAnalysis(new HashSet<Trace>(p.getTraces()));
        entanglementSubsets = entanglementAnalysis.getEntangledPartitions(subsetSize);
        entanglementScore = computeEntanglementScore(entanglementSubsets);
        System.out.println("Score is " + entanglementScore);
    }

    public int computeEntanglementScore(EntangledPartitions subsets) {
        int score = 0;
        for (Set<DynAngel> subset : subsets.entangledSubsets) {
            score += subset.size() * subset.size();
        }
        score *= 2;
        for (Set<DynAngel> subset : subsets.unentangledSubsets) {
            score += subset.size();
        }
        return score;
    }

    public int compareTo(PartitionInfo rhs) {
        if (entanglementScore < rhs.entanglementScore) {
            return -1;
        } else if (entanglementScore == rhs.entanglementScore) {
            return 0;
        } else {
            return 1;
        }
    }
}
