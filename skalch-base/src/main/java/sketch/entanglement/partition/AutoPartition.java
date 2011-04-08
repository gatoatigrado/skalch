package sketch.entanglement.partition;

import java.sql.Time;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class AutoPartition extends TracePartitioner {

    @Override
    public List<TraceSubset> getSubsets(TraceSubset p, String[] args) {
        if (args.length < 3) {
            ArrayList<TraceSubset> singletonList = new ArrayList<TraceSubset>();
            singletonList.add(p);
            return singletonList;
        }
        int depth = Integer.parseInt(args[0]);
        int branching = Integer.parseInt(args[1]);
        int subsetSize = Integer.parseInt(args[2]);

        BinaryHeap<PartitionInfo> curSolutionHeap = new BinaryHeap<PartitionInfo>();
        BinaryHeap<PartitionInfo> allSolutionHeap = new BinaryHeap<PartitionInfo>();

        HashMap<Integer, List<TraceSubset>> seenPartitions =
                new HashMap<Integer, List<TraceSubset>>();

        List<TraceSubset> newSolutionList = new ArrayList<TraceSubset>();
        newSolutionList.add(p);
        addToSeenPartitions(seenPartitions, p);

        for (int i = 0; i < depth; i++) {
            System.out.println("Depth " + i);

            for (TraceSubset newSolution : newSolutionList) {
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
                    List<TraceSubset> partitions =
                            partitioner.getSubsets(solution.partition, null);
                    // add each new partition to solutions list
                    for (TraceSubset partition : partitions) {
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

        ArrayList<TraceSubset> returnList = new ArrayList<TraceSubset>();
        while (curSolutionHeap.size() > 0) {
            returnList.add(curSolutionHeap.remove().partition);
        }
        returnList.addAll(newSolutionList);
        return returnList;
    }

    private boolean hasBeenSeenPartition(
            HashMap<Integer, List<TraceSubset>> seenPartitions,
            TraceSubset partition)
    {
        int size = partition.getTraces().size();
        if (seenPartitions.containsKey(size)) {
            for (TraceSubset sameSizedPartition : seenPartitions.get(size)) {
                if (partition.hasSameTraces(sameSizedPartition)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void addToSeenPartitions(
            HashMap<Integer, List<TraceSubset>> seenPartitions,
            TraceSubset partition)
    {
        int size = partition.getTraces().size();
        if (seenPartitions.containsKey(size)) {
            seenPartitions.get(size).add(partition);
        } else {
            List<TraceSubset> sameSizePartitions = new ArrayList<TraceSubset>();
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
        Set<Set<DynAngel>> es = info.entanglementSubsets;
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

        for (Set<DynAngel> unentangledSet : es) {
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
    public Set<Set<DynAngel>> entanglementSubsets;
    public final TraceSubset partition;
    public final int entanglementScore;

    public PartitionInfo(TraceSubset p, int subsetSize) {
        System.out.println("Calculating entanglement score for partition " +
                p.getPartitionName() + " (" + new Time(System.currentTimeMillis()) + ")");

        partition = p;
        entanglementAnalysis =
                new SATEntanglementAnalysis(new HashSet<Trace>(p.getTraces()));
        entanglementSubsets = entanglementAnalysis.getEntangledPartitions();
        entanglementScore = computeEntanglementScore(entanglementSubsets);
        System.out.println("Score is " + entanglementScore);
    }

    public int computeEntanglementScore(Set<Set<DynAngel>> entanglementSubsets) {
        int score = 0;
        for (Set<DynAngel> subset : entanglementSubsets) {
            score += subset.size() * subset.size();
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
