package sketch.entanglement.sat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

    public TraceConverter(Set<Trace> traces) {
        // A list version of the traces
        ArrayList<Trace> traceList = new ArrayList<Trace>(traces);
        // A mapping from dynamic angel to a list of integers. The ith value in the list
        // represents the value of the dynamic angel in the ith trace in traceList.
        Map<DynAngel, List<Integer>> tracesByAngelMap = getTracesByAngel(traceList);

        // sort the angels so they are in some order
        angels = new ArrayList<DynAngel>(tracesByAngelMap.keySet());
        Collections.sort(angels);

        simpleTraceToNormalTrace =
                getSimpleTraceMapping(traceList, tracesByAngelMap, angels);
        simpleTraces = simpleTraceToNormalTrace.keySet();

        maxValues = getMaxAngelValues(angels.size(), simpleTraces);

        assert simpleTraces.size() == traces.size();
    }

    // returns a mapping from a "ordered trace" to the original trace. ordered traces
    // contain values in a strict order and can also contains values for unencountered
    // angels (but encountered in another trace)
    private Map<List<Integer>, Trace> getSimpleTraceMapping(List<Trace> traceList,
            Map<DynAngel, List<Integer>> tracesByAngelMap, List<DynAngel> angelList)
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

        Map<List<Integer>, Trace> simpleTraceMapping =
                new HashMap<List<Integer>, Trace>();

        int numTraces = traceList.size();
        for (int i = 0; i < numTraces; i++) {
            List<Integer> trace = new ArrayList<Integer>();
            for (int j = 0; j < angelList.size(); j++) {
                DynAngel angel = angelList.get(j);
                // get the value of the angel at the ith trace
                int val = tracesByAngelMap.get(angel).get(i);
                int index = angelToValues.get(angel).indexOf(val);
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

    private Map<DynAngel, List<Integer>> getTracesByAngel(List<Trace> traces) {
        Map<DynAngel, List<Integer>> dynamicAngelsToValues =
                new HashMap<DynAngel, List<Integer>>();
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
                } else {
                    values = new ArrayList<Integer>();
                    dynamicAngelsToValues.put(dynAngel, values);
                }
                // if the dynamic angel was not accessed in the previous traces,
                // then we need to pad the list until the index
                while (values.size() < i) {
                    values.add(-1);
                }
                values.add(event.valueChosen);
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
}
