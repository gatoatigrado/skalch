package sketch.entanglement.sat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import kodkod.util.ints.IntBitSet;
import kodkod.util.ints.IntIterator;
import kodkod.util.ints.IntSet;
import kodkod.util.ints.Ints;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.Trace;
import entanglement.trace.Traces;

/**
 * Converts sketch.entanglement traces into traces that the SAT infrastructure can use. It
 * also translates these traces back to sketch.entanglement traces.
 * 
 * @author shaon
 */
public class TraceConverter {

    final private Set<List<Integer>> simpleTraces;
    final private int[] maxValues;

    final private Map<List<Integer>, Trace> simpleTraceToNormalTrace;
    final private ArrayList<DynAngel> angels;
    private Map<DynAngel, List<Integer>> angelsToValues;

    public TraceConverter(List<DynAngel> angels, Set<Trace> traces) {
        // A list version of the traces
        ArrayList<Trace> traceList = new ArrayList<Trace>(traces);
        // A mapping from dynamic angel to a list of integers. The ith value in the list
        // represents the value of the dynamic angel in the ith trace in traceList.
        Map<DynAngel, List<Integer>> tracesByAngelMap =
                getTracesByAngel(traceList, angels);
        this.angels = new ArrayList<DynAngel>(angels);
        angelsToValues = getAngelsToValues(tracesByAngelMap);

        simpleTraceToNormalTrace =
                getSimpleTraceMapping(traceList, tracesByAngelMap, angels, angelsToValues);
        simpleTraces = simpleTraceToNormalTrace.keySet();

        maxValues = getMaxAngelValues(angels.size(), simpleTraces);

        assert simpleTraces.size() == traces.size();
    }

    private Map<DynAngel, List<Integer>> getAngelsToValues(
            Map<DynAngel, List<Integer>> tracesByAngelMap)
    {
        Map<DynAngel, List<Integer>> angelToValues =
                new HashMap<DynAngel, List<Integer>>();
        for (DynAngel angel : tracesByAngelMap.keySet()) {
            Set<Integer> values = new HashSet<Integer>();
            for (Integer value : tracesByAngelMap.get(angel)) {
                values.add(value);
            }
            angelToValues.put(angel, new ArrayList<Integer>(values));
        }
        return angelToValues;
    }

    // returns a mapping from a "ordered trace" to the original trace. ordered traces
    // contain values in a strict order and can also contains values for unencountered
    // angels (but encountered in another trace)
    private Map<List<Integer>, Trace> getSimpleTraceMapping(List<Trace> traceList,
            Map<DynAngel, List<Integer>> tracesByAngelMap, List<DynAngel> angelList,
            Map<DynAngel, List<Integer>> angelsToValues)
    {

        Map<List<Integer>, Trace> simpleTraceMapping =
                new HashMap<List<Integer>, Trace>();

        int numTraces = traceList.size();
        for (int i = 0; i < numTraces; i++) {
            List<Integer> trace = new ArrayList<Integer>();
            for (int j = 0; j < angelList.size(); j++) {
                DynAngel angel = angelList.get(j);
                // get the value of the angel at the ith trace
                int val = tracesByAngelMap.get(angel).get(i);
                int index = angelsToValues.get(angel).indexOf(val);
                trace.add(index);
            }
            simpleTraceMapping.put(trace, traceList.get(i));
        }
        return simpleTraceMapping;
    }

    private int[] getMaxAngelValues(int numAngels, Set<List<Integer>> traces) {
        int max[] = new int[numAngels];
        for (List<Integer> trace : traces) {
            for (int i = 0; i < trace.size(); i++) {
                int val = trace.get(i);
                if (max[i] < val) {
                    max[i] = val;
                }
            }
        }
        return max;
    }

    private Map<DynAngel, List<Integer>> getTracesByAngel(List<Trace> traces,
            List<DynAngel> angels)
    {
        Map<DynAngel, List<Integer>> dynamicAngelsToValues =
                new HashMap<DynAngel, List<Integer>>();
        for (DynAngel angel : angels) {
            dynamicAngelsToValues.put(angel, new ArrayList<Integer>());
        }

        // go through every trace
        for (int i = 0; i < traces.size(); i++) {
            // go through every dynamic angelic call
            Trace trace = traces.get(i);
            for (Event event : trace.events) {
                DynAngel dynAngel = event.dynAngel;
                List<Integer> values;
                // add valueChosen to correct list
                if (dynamicAngelsToValues.containsKey(dynAngel)) {
                    values = dynamicAngelsToValues.get(dynAngel);

                    // if the dynamic angel was not accessed in the previous traces,
                    // then we need to pad the list until the index
                    while (values.size() < i) {
                        values.add(-1);
                    }
                    values.add(event.valueChosen);
                }
            }
        }
        // pad all the lists so they are the same size
        for (DynAngel dynamicAngel : dynamicAngelsToValues.keySet()) {
            List<Integer> values = dynamicAngelsToValues.get(dynamicAngel);
            while (values.size() < traces.size()) {
                values.add(-1);
            }
        }
        return dynamicAngelsToValues;
    }

    public Traces getTraces() {
        return Traces.traces(maxValues, simpleTraces);
    }

    public DynAngel getAngel(int index) {
        return angels.get(index);
    }

    public int getAngelValue(int index, int val) {
        DynAngel angel = getAngel(index);
        return angelsToValues.get(angel).get(val);
    }

    public List<DynAngel> getAngelList() {
        return new ArrayList<DynAngel>(angels);
    }

    public int getIndex(DynAngel d) {
        return angels.indexOf(d);
    }

    public List<Trace> convert(Traces traces) {
        List<Trace> normalTraces = new ArrayList<Trace>();
        for (Iterator<entanglement.trace.Trace> it = traces.iterator(); it.hasNext();) {
            entanglement.trace.Trace t = it.next();
            List<Integer> simpleTrace = new ArrayList<Integer>();
            for (int j = 0; j < t.length(); j++) {
                simpleTrace.add(t.get(j));
            }
            Trace normalTrace = simpleTraceToNormalTrace.get(simpleTrace);
            if (normalTrace == null) {
                throw new RuntimeException(
                        "Should not get a null trace in TraceConverter");
            } else {
                normalTraces.add(normalTrace);
            }
        }
        return normalTraces;
    }

    public Set<Set<DynAngel>> getDynAngelPartitions(List<IntSet> oldPartitions) {
        Set<Set<DynAngel>> dynAngelPartitions = new HashSet<Set<DynAngel>>();
        for (IntSet oldPartition : oldPartitions) {
            Set<DynAngel> partition = new HashSet<DynAngel>();
            for (IntIterator i = oldPartition.iterator(); i.hasNext();) {
                partition.add(getAngel(i.next()));
            }
            dynAngelPartitions.add(partition);
        }
        return dynAngelPartitions;
    }

    public List<IntSet> getIntSetPartitions(Set<Set<DynAngel>> partitioning) {
        List<IntSet> newPartitions = new ArrayList<IntSet>();

        for (Set<DynAngel> partition : partitioning) {
            if (partition.size() == 1) {
                newPartitions.add(Ints.singleton(getIndex(partition.iterator().next())));
            } else {
                int maxValue = -1;
                List<Integer> indexes = new ArrayList<Integer>();
                for (DynAngel angel : partition) {
                    int index = getIndex(angel);
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
        return newPartitions;
    }
}
