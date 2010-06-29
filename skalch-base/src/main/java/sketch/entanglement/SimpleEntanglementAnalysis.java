package sketch.entanglement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SimpleEntanglementAnalysis {

    private final List<Trace> traces;
    // private final Set<DynAngelPair> entangledAngelPairs;
    private final Set<DynAngel> angels;
    private final Map<Set<DynAngel>, Set<Trace>> projValues;

    public SimpleEntanglementAnalysis(Collection<Trace> traces) {
        this.traces = new ArrayList<Trace>(traces);
        projValues = new HashMap<Set<DynAngel>, Set<Trace>>();
        angels = findAngels();
        // entangledAngelPairs = findEntangledPairs();
    }

    private Set<DynAngel> findAngels() {
        HashSet<DynAngel> angelSet = new HashSet<DynAngel>();
        for (Trace trace : traces) {
            for (Event event : trace.getEvents()) {
                angelSet.add(event.dynAngel);
            }
        }
        return angelSet;
    }

    // private Set<DynAngelPair> findEntangledPairs() {
    // // list of all dynamic angels
    // ArrayList<DynAngel> dynamicAngels = new ArrayList<DynAngel>();
    // dynamicAngels.addAll(angels);
    // HashSet<DynAngelPair> entangledPairs = new HashSet<DynAngelPair>();
    //
    // for (int i = 0; i < dynamicAngels.size(); i++) {
    // for (int j = i + 1; j < dynamicAngels.size(); j++) {
    //
    // Set<DynAngel> proj1 = new HashSet<DynAngel>();
    // proj1.add(dynamicAngels.get(i));
    //
    // Set<DynAngel> proj2 = new HashSet<DynAngel>();
    // proj2.add(dynamicAngels.get(j));
    //
    // if (isEntangled(proj1, proj2)) {
    // entangledPairs.add(new DynAngelPair(dynamicAngels.get(i),
    // dynamicAngels.get(j)));
    // }
    // }
    // }
    // return entangledPairs;
    // }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getEntangledPairs()
     */
    // public Set<DynAngelPair> getEntangledPairs() {
    // return new HashSet<DynAngelPair>(entangledAngelPairs);
    // }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#isEntangled(java.util.Set,
     * java.util.Set)
     */
    public boolean isEntangled(Set<DynAngel> proj1, Set<DynAngel> proj2) {
        return entangledInfo(proj1, proj2, false).isEntangled;
    }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#compareTwoSubtraces(java.util.Set,
     * java.util.Set, boolean)
     */
    public EntanglementComparison entangledInfo(Set<DynAngel> proj1, Set<DynAngel> proj2,
            boolean verbose)
    {

        List<Trace> proj1TraceList = new ArrayList<Trace>(getValues(proj1));
        HashMap<Trace, Integer> proj1ToIndex = new HashMap<Trace, Integer>();
        for (int i = 0; i < proj1TraceList.size(); i++) {
            proj1ToIndex.put(proj1TraceList.get(i), i);
        }

        List<Trace> proj2TraceList = new ArrayList<Trace>(getValues(proj2));
        HashMap<Trace, Integer> proj2ToIndex = new HashMap<Trace, Integer>();
        for (int i = 0; i < proj2TraceList.size(); i++) {
            proj2ToIndex.put(proj2TraceList.get(i), i);
        }

        // default value is 0
        int[][] correlationMap = new int[proj1TraceList.size()][proj2TraceList.size()];

        if (verbose) {
            for (Trace trace : traces) {
                int i = proj1ToIndex.get(trace.getSubTrace(proj1));
                int j = proj2ToIndex.get(trace.getSubTrace(proj2));

                correlationMap[i][j]++;
            }
        } else {
            int numSeen = 0;
            int numGoal = proj1TraceList.size() * proj2TraceList.size();

            for (Trace trace : traces) {
                int i = proj1ToIndex.get(trace.getSubTrace(proj1));
                int j = proj2ToIndex.get(trace.getSubTrace(proj2));

                if (correlationMap[i][j] == 0) {
                    correlationMap[i][j] = 1;
                    numSeen++;
                    if (numSeen == numGoal) {
                        break;
                    }
                }
            }
        }

        EntanglementComparison ec =
                new EntanglementComparison(proj1, proj2, proj1TraceList, proj2TraceList,
                        correlationMap);
        return ec;
    }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getConstantAngels()
     */
    public Set<DynAngel> getConstantAngels() {
        HashMap<DynAngel, Boolean> isConstantAngel = new HashMap<DynAngel, Boolean>();
        HashMap<DynAngel, Integer> angelValues = new HashMap<DynAngel, Integer>();

        // go through every trace and event
        for (Trace trace : traces) {
            for (Event event : trace.events) {
                DynAngel location = event.dynAngel;
                // if the location is already visited and is said to be constant
                if (isConstantAngel.keySet().contains(location)) {
                    // check if location is still constant
                    if (isConstantAngel.get(location) &&
                            !angelValues.get(location).equals(event.valueChosen))
                    {
                        isConstantAngel.put(location, false);
                    }
                } else {
                    isConstantAngel.put(location, true);
                    angelValues.put(location, event.valueChosen);
                }
            }
        }
        Set<DynAngel> constantAngels = new HashSet<DynAngel>();
        for (DynAngel angel : isConstantAngel.keySet()) {
            if (isConstantAngel.get(angel)) {
                constantAngels.add(angel);
            }
        }
        return constantAngels;
    }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getEntangledAngels()
     */
    // public Set<DynAngel> getEntangledAngels() {
    // HashSet<DynAngel> entangledAngels = new HashSet<DynAngel>();
    // for (DynAngelPair pair : entangledAngelPairs) {
    // entangledAngels.add(pair.getFirst());
    // entangledAngels.add(pair.getSecond());
    // }
    // return entangledAngels;
    // }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getAngels()
     */
    public Set<DynAngel> getAngels() {
        return angels;
    }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getPossibleValues(java.util.Set)
     */
    public Set<Trace> getValues(Set<DynAngel> proj) {
        if (projValues.containsKey(proj)) {
            return new HashSet<Trace>(projValues.get(proj));
        }

        Set<Trace> projTraceSet = new HashSet<Trace>();
        for (Trace trace : traces) {
            projTraceSet.add(trace.getSubTrace(proj));
        }
        projValues.put(new HashSet<DynAngel>(proj), new HashSet<Trace>(projTraceSet));
        return projTraceSet;
    }

    // private Set<Set<DynAngel>> getOneEntangledSubsets() {
    // ScGraph<DynAngel> entanglementGraph = new ScGraph<DynAngel>();
    // for (DynAngel angel : angels) {
    // entanglementGraph.addVertex(angel);
    // }
    //
    // for (DynAngelPair entangledAngels : entangledAngelPairs) {
    // entanglementGraph.addEdge(entangledAngels.loc1, entangledAngels.loc2);
    // }
    // return entanglementGraph.getConnectedComponents();
    // }

    /*
     * (non-Javadoc)
     * @see sketch.entanglement.IEntanglementAnalysis#getNEntangledSubsets(int)
     */
    // public EntangledPartitions getEntangledPartitions(int n) {
    // List<Set<DynAngel>> entangledSubsets =
    // new ArrayList<Set<DynAngel>>(getOneEntangledSubsets());
    // Set<Set<DynAngel>> unentangledSubsets = new HashSet<Set<DynAngel>>();
    //
    // Set<DynAngel> entangledAngels = new HashSet<DynAngel>(angels);
    //
    // for (Set<DynAngel> subset : entangledSubsets) {
    // if (!isBetaEntangled(subset, entangledAngels)) {
    // unentangledSubsets.add(subset);
    // entangledAngels.removeAll(subset);
    // }
    // }
    // entangledSubsets.removeAll(unentangledSubsets);
    //
    // HashSet<Integer> ignoreList = new HashSet<Integer>();
    // SubsetIterator iterator = new SubsetIterator(2, n, entangledSubsets.size());
    // subsets: for (; iterator.hasNext();) {
    // List<Integer> subsetIndexes = iterator.next();
    // HashSet<DynAngel> unionSubset = new HashSet<DynAngel>();
    // for (Integer index : subsetIndexes) {
    // if (ignoreList.contains(index)) {
    // continue subsets;
    // }
    // unionSubset.addAll(entangledSubsets.get(index));
    // }
    //
    // if (!isBetaEntangled(unionSubset, entangledAngels)) {
    // unentangledSubsets.add(new HashSet<DynAngel>(unionSubset));
    // entangledAngels.removeAll(unionSubset);
    // ignoreList.addAll(subsetIndexes);
    // if (ignoreList.size() == entangledSubsets.size()) {
    // break;
    // }
    // }
    // }
    //
    // List<Set<DynAngel>> subsetsToRemove = new ArrayList<Set<DynAngel>>();
    //
    // for (Integer index : ignoreList) {
    // subsetsToRemove.add(entangledSubsets.get(index));
    // }
    // entangledSubsets.removeAll(subsetsToRemove);
    // return new EntangledPartitions(unentangledSubsets, new HashSet<Set<DynAngel>>(
    // entangledSubsets));
    // }

    // private boolean isBetaEntangled(Set<DynAngel> subset, Set<DynAngel>
    // entangledAngels) {
    // Set<DynAngel> complementAngels = new HashSet<DynAngel>(entangledAngels);
    // complementAngels.removeAll(subset);
    // return isEntangled(subset, complementAngels);
    // }
}
