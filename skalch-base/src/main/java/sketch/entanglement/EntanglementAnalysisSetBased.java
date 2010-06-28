package sketch.entanglement;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class EntanglementAnalysisSetBased {

    final private List<Trace> traces;
    final private Map<DynAngel, List<Integer>> angelsToValueMap;
    final private Map<DynAngel, List<Integer>> angelsToValues;
    final private EntanglementAnalysis ea;

    final private Random rand;

    public EntanglementAnalysisSetBased(Collection<Trace> traces) {
        ea = null;// new EntanglementAnalysis(traces);
        angelsToValueMap = getAngelsToValuesMap(new ArrayList<Trace>(traces));
        this.traces = getSetBasedTraces(angelsToValueMap);
        angelsToValues = getAngelsToValues(angelsToValueMap);
        rand = new Random();
    }

    private List<Trace> getSetBasedTraces(Map<DynAngel, List<Integer>> angelsToValueMap) {
        List<Trace> setBasedTraces = new ArrayList<Trace>();
        List<DynAngel> angels = new ArrayList<DynAngel>(angelsToValueMap.keySet());
        Collections.sort(angels);
        int numTraces = angelsToValueMap.get(angels.get(0)).size();
        for (int i = 0; i < numTraces; i++) {
            List<Event> events = new ArrayList<Event>();
            for (int j = 0; j < angels.size(); j++) {
                DynAngel angel = angels.get(j);
                int val = angelsToValueMap.get(angel).get(i);
                events.add(new Event(angel.staticAngelId, angel.execNum, 0, val));
            }
            setBasedTraces.add(new Trace(events));
        }
        return setBasedTraces;
    }

    private Map<DynAngel, List<Integer>> getAngelsToValuesMap(List<Trace> traces) {
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

    private Map<DynAngel, List<Integer>> getAngelsToValues(
            Map<DynAngel, List<Integer>> tracesByAngels)
    {
        Map<DynAngel, List<Integer>> angelsToValues =
                new HashMap<DynAngel, List<Integer>>();
        for (DynAngel angel : tracesByAngels.keySet()) {
            Set<Integer> values = new HashSet<Integer>(tracesByAngels.get(angel));
            angelsToValues.put(angel, new ArrayList<Integer>(values));
        }
        return angelsToValues;
    }

    public Set<DynAngel> findEntanglement() {
        Trace startTrace = null;
        while (startTrace == null) {
            int startTraceIndex = rand.nextInt(traces.size());
            Trace t = traces.get(startTraceIndex);
            List<Event> events = t.getEvents();
            int index = rand.nextInt(events.size());
            Event e = events.get(index);
            int newVal = rand.nextInt(angelsToValues.get(e.dynAngel).size());
            Trace editedTrace = getEditedTrace(t, index, newVal);
            if (!traces.contains(editedTrace)) {
                startTrace = editedTrace;
            }
        }
        System.out.println("Found start trace");
        Set<Trace> curTraces = new HashSet<Trace>();
        curTraces.add(startTrace);

        Set<DynAngel> entangled = new HashSet<DynAngel>();
        List<DynAngel> unseenAngels = new ArrayList<DynAngel>(angelsToValues.keySet());
        while (!unseenAngels.isEmpty()) {
            int angelIndex = rand.nextInt(unseenAngels.size());
            System.out.println("On index: " + angelIndex);
            System.out.print("Entangled: ");
            for (DynAngel e : entangled) {
                System.out.print(e + ";");
            }
            System.out.println();
            DynAngel angel = unseenAngels.remove(angelIndex);
            List<Integer> values = angelsToValues.get(angel);
            boolean isEntangled = false;

            List<Trace> newTraces1 = new ArrayList<Trace>();
            for (Integer value : values) {
                boolean canExpand = true;
                List<Trace> newTraces2 = new ArrayList<Trace>();
                trace: for (Trace trace : curTraces) {
                    Trace editedTrace = getEditedTrace(trace, angelIndex, value);
                    if (traces.contains(editedTrace)) {
                        isEntangled = true;
                        canExpand = false;
                        break trace;
                    }
                    newTraces2.add(editedTrace);
                }
                if (canExpand) {
                    newTraces1.addAll(newTraces2);
                }
            }

            if (isEntangled) {
                entangled.add(angel);
            }
            curTraces.addAll(newTraces1);
        }

        return entangled;
    }

    private Trace getEditedTrace(Trace t, int index, int newVal) {
        List<Event> initTrace = t.getEvents();
        Event event = initTrace.get(index);
        Event newEvent =
                new Event(event.dynAngel.staticAngelId, event.dynAngel.execNum,
                        event.numValues, newVal);
        initTrace.set(index, newEvent);
        return new Trace(initTrace);
    }

    public void printBitStrings(String fileName) {
        FileOutputStream fout = null;
        PrintStream out = null;

        try {
            fout = new FileOutputStream(fileName);
            out = new PrintStream(fout);
            for (Trace t : traces) {
                List<Event> events = t.getEvents();
                StringBuilder bitString = new StringBuilder();
                for (int i = 0; i < events.size(); i++) {
                    Event e = events.get(i);
                    List<Integer> values = angelsToValues.get(e.dynAngel);
                    int value = values.indexOf(e.valueChosen);

                    assert value != -1;

                    bitString.append(value);
                    if (i != events.size() - 1) {
                        bitString.append(',');
                    }
                }
                out.println(bitString.toString());
            }

        } catch (FileNotFoundException e) {
            System.err.println("Writing out bistring caused FileNotFoundException.");
        } finally {
            if (out != null) {
                out.close();
            }
        }

        try {
            fout = new FileOutputStream(fileName + "-info");
            out = new PrintStream(fout);
            List<Event> events = traces.get(0).getEvents();
            List<DynAngel> dynAngels = new ArrayList<DynAngel>();
            for (int i = 0; i < events.size(); i++) {
                dynAngels.add(events.get(i).dynAngel);
            }

            for (int i = 0; i < dynAngels.size(); i++) {
                DynAngel dynAngel = dynAngels.get(i);
                List<Integer> values = angelsToValues.get(dynAngel);
                StringBuilder eventString = new StringBuilder(i + ": " + dynAngel + "{");
                for (int j = 0; j < values.size(); j++) {
                    eventString.append(values.get(j));
                    if (j != values.size() - 1) {
                        eventString.append(',');
                    }
                }
                eventString.append('}');
                out.println(eventString.toString());
            }
            out.println();
            EntanglementSubsets subsets = ea.getNEntangledSubsets(dynAngels.size());
            out.println("Entangled");
            for (Set<DynAngel> subset : subsets.entangledSubsets) {
                List<DynAngel> subsetList = new ArrayList<DynAngel>(subset);
                Collections.sort(subsetList);
                StringBuilder dynAngelString = new StringBuilder();
                for (int i = 0; i < subsetList.size(); i++) {
                    dynAngelString.append(dynAngels.indexOf(subsetList.get(i)));
                    if (i != subsetList.size() - 1) {
                        dynAngelString.append(',');
                    }
                }
                out.println(dynAngelString.toString());
            }

            out.println("Unentangled");
            for (Set<DynAngel> subset : subsets.unentangledSubsets) {
                List<DynAngel> subsetList = new ArrayList<DynAngel>(subset);
                Collections.sort(subsetList);
                StringBuilder dynAngelString = new StringBuilder();
                for (int i = 0; i < subsetList.size(); i++) {
                    dynAngelString.append(dynAngels.indexOf(subsetList.get(i)));
                    if (i != subsetList.size() - 1) {
                        dynAngelString.append(',');
                    }
                }
                out.println(dynAngelString.toString());
            }

        } catch (FileNotFoundException e) {
            System.err.println("Writing out bistring caused FileNotFoundException.");
        } finally {
            if (out != null) {
                out.close();
            }
        }

    }
}
