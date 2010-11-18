package sketch.entanglement;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import kodkod.util.ints.IntSet;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.entanglement.sat.TraceConverter;
import entanglement.MaxSupportFinder;
import entanglement.trace.Traces;

public class HeuristicSearch {

    class TraceSubset implements Comparable<TraceSubset> {

        public final int value;
        public final Set<Trace> subset;

        public TraceSubset(Set<Trace> subset) {
            this.subset = subset;

            SimpleEntanglementAnalysis ea = new SimpleEntanglementAnalysis(subset);
            SATEntanglementAnalysis satEA = new SATEntanglementAnalysis(subset);

            Set<DynAngel> constant = ea.getConstantAngels();

            float numConstantStaticAngels = getNumConstantStaticAngels(ea, subset);

            int numPermutations = getNumPermutations(ea);

            Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();
            int entanglementScore =
                    getEntanglementScore(
                            HeuristicSearch.this.satEA.getEntangledPartitions(),
                            partitions);

            value =
                    getScore(constant.size(), numConstantStaticAngels, numPermutations,
                            entanglementScore);
        }

        public int compareTo(TraceSubset other) {
            if (value > other.value) {
                return -1;
            } else if (value < other.value) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    private SATEntanglementAnalysis satEA;
    private SimpleEntanglementAnalysis ea;
    private PriorityQueue<TraceSubset> rankedTraceSubsets;
    private Set<Set<Trace>> subsetsSet;
    private List<Set<Trace>> subsetsList;
    private List<DynAngel> angelOrder;

    public HeuristicSearch(List<DynAngel> angelOrder, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        this.angelOrder = angelOrder;
        this.satEA = satEA;
        this.ea = ea;

        subsetsSet = new HashSet<Set<Trace>>();
        subsetsList = new ArrayList<Set<Trace>>();
        rankedTraceSubsets = new PriorityQueue<TraceSubset>();
    }

    public void partition() {
        Set<Set<DynAngel>> oldPartitions = satEA.getEntangledPartitions();

        for (int i = 1; i < angelOrder.size(); i++) {

            DynAngel pivotAngel = angelOrder.get(i);
            Set<DynAngel> pivotAngelPartition = null;
            for (Set<DynAngel> partition : oldPartitions) {
                if (partition.contains(pivotAngel)) {
                    pivotAngelPartition = partition;
                    break;
                }
            }
            if (pivotAngelPartition.size() == 1) {
                continue;
            }

            Set<DynAngel> firstPartition =
                    new HashSet<DynAngel>(angelOrder.subList(0, i));
            Set<DynAngel> secondPartition =
                    new HashSet<DynAngel>(angelOrder.subList(i, angelOrder.size()));

            Set<Set<DynAngel>> subpartitioning = new HashSet<Set<DynAngel>>();
            subpartitioning.add(firstPartition);
            subpartitioning.add(secondPartition);

            Set<Set<DynAngel>> partitioning =
                    getNewPartitioning(subpartitioning, oldPartitions);
            TraceConverter converter = satEA.getTraceConverter();
            List<IntSet> newPartitions = converter.getIntSetPartitions(partitioning);

            Traces satTraces = converter.getTraces();

            Set<Set<Trace>> subsets = new HashSet<Set<Trace>>();

            for (Iterator<Traces> supports =
                    MaxSupportFinder.findMaximalSupports(satTraces,
                            satEA.getEntangledIntSets(), newPartitions); supports.hasNext();)
            {
                Traces support = supports.next();
                List<Trace> subsetTraces = converter.convert(support);
                HashSet<Trace> subsetTracesSet = new HashSet<Trace>(subsetTraces);
                subsets.add(subsetTracesSet);
                subsetsSet.add(subsetTracesSet);
                subsetsList.add(subsetTracesSet);
            }

            System.out.println("Partition around " + i + "(" + angelOrder.get(i) +
                    ") : " + subsets.size());

            for (Set<Trace> subset : subsets) {
                analyzeSubset(subset);
            }

        }
        rankAllSubsets();

        System.out.println("Number LMS: " + subsetsList.size());
        System.out.println("Number unique LMS: " + subsetsSet.size());
    }

    private Set<Set<DynAngel>> getNewPartitioning(Set<Set<DynAngel>> subpartitioning,
            Set<Set<DynAngel>> oldPartitions)
    {
        Set<Set<DynAngel>> partitioning = new HashSet<Set<DynAngel>>();

        for (Set<DynAngel> partition : oldPartitions) {
            HashSet<DynAngel> partitionClone = new HashSet<DynAngel>(partition);
            for (Set<DynAngel> subpartition : subpartitioning) {
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
        return partitioning;
    }

    private void analyzeSubset(Set<Trace> subset) {
        System.out.print(subset.size() + "\t");

        int maxSize = 0;
        int minSize = Integer.MAX_VALUE;

        for (Trace trace : subset) {
            int size = trace.size();
            if (maxSize < size) {
                maxSize = size;
            }
            if (minSize > size) {
                minSize = size;
            }
        }
        System.out.print(", (" + minSize + "," + maxSize + ")\t");

        SimpleEntanglementAnalysis ea = new SimpleEntanglementAnalysis(subset);
        SATEntanglementAnalysis satEA = new SATEntanglementAnalysis(subset);

        Set<DynAngel> constant = ea.getConstantAngels();
        int numConstantDynamicAngels = constant.size();
        System.out.print(numConstantDynamicAngels + "\t");

        float numConstantStaticAngels = getNumConstantStaticAngels(ea, subset);
        System.out.print(numConstantStaticAngels + "\t");

        int numPermutations = getNumPermutations(ea);
        System.out.print(numPermutations + "\t");

        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();

        int numPermutationsInEntanglement =
                getNumPermutationsInEntanglement(ea, partitions);
        System.out.print(numPermutationsInEntanglement + "\t");

        int entanglementScore =
                getEntanglementScore(this.satEA.getEntangledPartitions(), partitions);
        System.out.print(entanglementScore + "\t");

        int score =
                getScore(numConstantDynamicAngels, numConstantStaticAngels,
                        numPermutations, entanglementScore);
        System.out.print(score + "\t");

        System.out.print("(");
        for (Set<DynAngel> partition : partitions) {
            System.out.print(partition.size() + ",");
        }
        System.out.print(")");

        System.out.println();
    }

    private int getScore(int numDynamicConstantAngels, float numConstantStaticAngels,
            float numPermutations, int entanglementScore)
    {
        return (int) ((numConstantStaticAngels + numPermutations) * 1000 + entanglementScore);
    }

    public static int getEntanglementScore(Set<Set<DynAngel>> oldPartitions,
            Set<Set<DynAngel>> partitions)
    {
        int base = getEntanglementScore(oldPartitions);
        int current = getEntanglementScore(partitions);

        return (int) Math.round(10.0 * base / current);
    }

    public static int getEntanglementScore(Set<Set<DynAngel>> partitions) {
        int score = 1;
        int totalAngels = 0;
        for (Set<DynAngel> partition : partitions) {
            score += partition.size() * partition.size();
            totalAngels += partition.size();
        }

        return score - totalAngels;
    }

    public static float getNumConstantStaticAngels(SimpleEntanglementAnalysis ea,
            Set<Trace> subset)
    {
        Map<Integer, Set<DynAngel>> staticAngels = getStaticAngelMapping(ea);

        float maxConstantStaticAngels = 0;

        for (Trace trace : subset) {
            float numConstantStaticAngels = 0;
            for (Integer staticId : staticAngels.keySet()) {
                Set<DynAngel> staticAngelSet = staticAngels.get(staticId);
                Trace value = trace.getSubTrace(staticAngelSet);
                List<Event> events = value.events;
                if (events.isEmpty()) {
                    continue;
                }
                // System.out.print(events);

                Map<Integer, Integer> values = new HashMap<Integer, Integer>();
                for (Event event : events) {
                    int valueChosen = event.valueChosen;
                    if (values.containsKey(valueChosen)) {
                        values.put(valueChosen, values.get(valueChosen) + 1);
                    } else {
                        values.put(valueChosen, 1);
                    }
                }

                int maxOccurences = 0;
                for (Integer valueChosen : values.keySet()) {
                    int numOccurences = values.get(valueChosen);
                    if (numOccurences > maxOccurences) {
                        maxOccurences = numOccurences;

                    }
                }
                numConstantStaticAngels += ((float) maxOccurences) / events.size();
                // System.out.println("  " + numConstantStaticAngels);
            }
            // System.out.println("  " + numConstantStaticAngels);
            if (numConstantStaticAngels > maxConstantStaticAngels) {
                maxConstantStaticAngels = numConstantStaticAngels;
            }
        }
        return maxConstantStaticAngels;
    }

    public static int getNumPermutationsInEntanglement(SimpleEntanglementAnalysis ea,
            Set<Set<DynAngel>> partitions)
    {
        int numPermutationAngels = 0;

        for (Set<DynAngel> partition : partitions) {
            Set<Trace> values = ea.getValues(partition);
            boolean isPermutation = true;
            for (Trace value : values) {
                List<Event> events = value.events;
                if (events.isEmpty()) {
                    continue;
                }
                Set<Integer> valuesSeen = new HashSet<Integer>();
                for (Event event : events) {
                    if (valuesSeen.contains(event.valueChosen)) {
                        isPermutation = false;
                        break;
                    }
                    valuesSeen.add(event.valueChosen);
                }
            }
            if (isPermutation) {
                numPermutationAngels++;
            }
        }
        return numPermutationAngels;
    }

    public static int getNumPermutations(SimpleEntanglementAnalysis ea) {
        Map<Integer, Set<DynAngel>> staticAngels = getStaticAngelMapping(ea);

        int numPermutationAngels = 0;

        for (Integer staticId : staticAngels.keySet()) {
            Set<DynAngel> staticAngelSet = staticAngels.get(staticId);
            Set<Trace> values = ea.getValues(staticAngelSet);
            boolean isPermutation = true;
            for (Trace value : values) {
                List<Event> events = value.events;
                if (events.isEmpty()) {
                    continue;
                }
                Set<Integer> valuesSeen = new HashSet<Integer>();
                for (Event event : events) {
                    if (valuesSeen.contains(event.valueChosen)) {
                        isPermutation = false;
                        break;
                    }
                    valuesSeen.add(event.valueChosen);
                }
            }
            if (isPermutation) {
                numPermutationAngels++;
            }
        }
        return numPermutationAngels;
    }

    private static Map<Integer, Set<DynAngel>> getStaticAngelMapping(
            SimpleEntanglementAnalysis ea)
    {
        Set<DynAngel> angels = ea.getAngels();
        Map<Integer, Set<DynAngel>> staticAngels = new HashMap<Integer, Set<DynAngel>>();

        for (DynAngel angel : angels) {
            int staticId = angel.staticAngelId;
            if (staticAngels.containsKey(staticId)) {
                staticAngels.get(staticId).add(angel);
            } else {
                Set<DynAngel> angelSet = new HashSet<DynAngel>();
                angelSet.add(angel);
                staticAngels.put(staticId, angelSet);
            }
        }
        return staticAngels;
    }

    public void rankAllSubsets() {
        for (Set<Trace> subset : subsetsSet) {
            rankedTraceSubsets.add(new TraceSubset(subset));
        }
    }

    public Set<Set<Trace>> getSubsetsSet() {
        return subsetsSet;
    }

    public List<Set<Trace>> getSubsetsList() {
        return subsetsList;
    }

    public List<Set<Trace>> getTopSubsets(int numSubsets) {
        List<Set<Trace>> subsets = new ArrayList<Set<Trace>>();
        PriorityQueue<TraceSubset> allSubsets =
                new PriorityQueue<TraceSubset>(rankedTraceSubsets);
        int i = 0;
        while (subsets.size() < numSubsets && !allSubsets.isEmpty()) {
            Set<Trace> removedSubset = allSubsets.remove().subset;
            analyzeSubset(removedSubset);
            subsets.add(new HashSet<Trace>(removedSubset));
            i++;
        }
        return subsets;
    }
}
