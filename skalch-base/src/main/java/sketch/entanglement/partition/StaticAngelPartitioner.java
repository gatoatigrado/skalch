package sketch.entanglement.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import sketch.entanglement.Event;
import sketch.entanglement.Trace;

public class StaticAngelPartitioner extends TracePartitioner {

    @Override
    public List<SubsetOfTraces> getSubsets(SubsetOfTraces p, String args[]) {
        List<Trace> traces = p.getTraces();

        int staticAngel;
        if (args.length == 1) {
            staticAngel = Integer.parseInt(args[0]);
        } else {
            ArrayList<SubsetOfTraces> returnPartition = new ArrayList<SubsetOfTraces>();
            return returnPartition;
        }

        HashMap<Integer, List<Trace>> valuePartitions =
                new HashMap<Integer, List<Trace>>();
        int noSameValue = -1;
        valuePartitions.put(noSameValue, new ArrayList<Trace>());

        for (Trace trace : traces) {
            int value = -1;
            boolean sameValue = true;
            for (Event event : trace.getEvents()) {
                if (event.dynAngel.staticAngelId == staticAngel) {
                    if (value == -1) {
                        value = event.valueChosen;
                    } else if (value != event.valueChosen) {
                        sameValue = false;
                        break;
                    }
                }
            }
            if (sameValue) {
                if (!valuePartitions.containsKey(value)) {
                    valuePartitions.put(value, new ArrayList<Trace>());
                }
                valuePartitions.get(value).add(trace);
            } else {
                valuePartitions.get(noSameValue).add(trace);
            }
        }

        List<SubsetOfTraces> partitions = new ArrayList<SubsetOfTraces>();
        if (!valuePartitions.get(noSameValue).isEmpty()) {
            SubsetOfTraces partition =
                    new SubsetOfTraces(valuePartitions.get(noSameValue), "sa" + staticAngel +
                            "=diff", p);
            partitions.add(partition);
        }
        valuePartitions.remove(noSameValue);
        for (Integer value : valuePartitions.keySet()) {
            SubsetOfTraces partition =
                    new SubsetOfTraces(valuePartitions.get(value), "sa" + staticAngel + "=" +
                            value, p);
            partitions.add(partition);
        }
        return partitions;
    }

    @Override
    public String toString() {
        // TODO Auto-generated method stub
        return "Static Angel Partitioner";
    }
}
