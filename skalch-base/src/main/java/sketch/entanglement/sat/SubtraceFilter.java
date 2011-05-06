package sketch.entanglement.sat;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import entanglement.trace.Trace;
import entanglement.trace.TraceFilter;

public class SubtraceFilter implements TraceFilter {

    final private TraceConverter converter;
    final private HashSet<Map<DynAngel, Integer>> convertedTraces;
    private boolean negate;

    public SubtraceFilter(Set<sketch.entanglement.Trace> traces,
            TraceConverter converter, boolean negate)
    {
        convertedTraces = new HashSet<Map<DynAngel, Integer>>();
        this.negate = negate;

        for (sketch.entanglement.Trace trace : traces) {
            Map<DynAngel, Integer> events = new HashMap<DynAngel, Integer>();
            for (Event event : trace.getEvents()) {
                events.put(event.dynAngel, event.valueChosen);
            }
            convertedTraces.add(events);
        }
        this.converter = converter;
    }

    public boolean accept(Trace trace) {
        boolean exists = existsInSubset(trace);
        if (!negate) {
            return exists;
        } else {
            return !exists;
        }
    }

    public boolean existsInSubset(Trace t) {
        System.out.println(t);
        trace: for (Map<DynAngel, Integer> convertedTrace : convertedTraces) {
            String output = "";
            for (int i = 0; i < t.length(); i++) {
                DynAngel angel = converter.getAngel(i);
                int angelValue = converter.getAngelValue(i, t.get(i));
                output += angel + "=" + angelValue + ", ";
                if (angelValue == -1) {
                    if (convertedTrace.containsKey(angel)) {
                        System.out.println(convertedTrace);
                        System.out.println(output);
                        continue trace;
                    }
                } else {
                    if (convertedTrace.containsKey(angel) &&
                            convertedTrace.get(angel) != angelValue)
                    {
                        System.out.println(convertedTrace);
                        System.out.println(output);
                        continue trace;
                    }
                }
            }
            return true;
        }
        return false;
    }
}
