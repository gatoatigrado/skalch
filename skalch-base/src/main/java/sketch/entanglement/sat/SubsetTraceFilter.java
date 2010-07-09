package sketch.entanglement.sat;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import entanglement.trace.Trace;
import entanglement.trace.TraceFilter;

public class SubsetTraceFilter implements TraceFilter {

    final private TraceConverter converter;
    final private HashSet<Map<DynAngel, Integer>> convertedTraces;

    public SubsetTraceFilter(Set<sketch.entanglement.Trace> traces,
            TraceConverter converter)
    {
        convertedTraces = new HashSet<Map<DynAngel, Integer>>();

        for (sketch.entanglement.Trace trace : traces) {
            Map<DynAngel, Integer> events = new HashMap<DynAngel, Integer>();
            for (Event event : trace.getEvents()) {
                events.put(event.dynAngel, event.valueChosen);
            }
            convertedTraces.add(events);
        }
        this.converter = converter;
    }

    public boolean accept(Trace t) {
        trace: for (Map<DynAngel, Integer> convertedTrace : convertedTraces) {
            for (int i = 0; i < t.length(); i++) {
                DynAngel angel = converter.getAngel(i);
                int angelValue = converter.getAngelValue(i, t.get(i));
                if (angelValue == -1) {
                    if (convertedTrace.containsKey(angel)) {
                        continue trace;
                    }
                } else {
                    if (!convertedTrace.containsKey(angel) ||
                            convertedTrace.get(angel) != angelValue)
                    {
                        continue trace;
                    }
                }
            }
            return true;
        }
        return false;
    }
}
