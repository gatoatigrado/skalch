package sketch.ui.entanglement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import sketch.util.DebugOut;

public class EntanglementAnalysis {

    private List<ExecutionTrace> traces;

    public EntanglementAnalysis(List<ExecutionTrace> traces) {
        this.traces = traces;
    }

    public void findDontCareElements() {
        HashMap<AngelicCallLoc, Boolean> constantCalls = new HashMap<AngelicCallLoc, Boolean>();
        HashMap<AngelicCallLoc, Integer> constantValues = new HashMap<AngelicCallLoc, Integer>();

        // go through every trace
        for (ExecutionTrace trace : traces) {
            // go through every angelic call
            for (AngelicCallInfo angelicCall : trace.angelicCalls) {
                AngelicCallLoc location = angelicCall.location;
                // if the location is already visited and is said to be constant
                if (constantCalls.keySet().contains(location)
                        && constantCalls.get(location)) {
                    // check if location is still constant
                    if (!constantValues.get(location).equals(
                            angelicCall.valueChosen)) {
                        constantCalls.put(location, false);
                    }
                } else {
                    // check if first time visiting location
                    if (!constantCalls.keySet().contains(location)) {
                        constantCalls.put(location, true);
                        constantValues.put(location, angelicCall.valueChosen);
                    }
                }

            }
        }

        List<AngelicCallLoc> locations = new ArrayList<AngelicCallLoc>(
                constantValues.keySet());
        Collections.sort(locations);
        DebugOut.print("Constant elements");
        for (AngelicCallLoc location : locations) {
            if (constantCalls.get(location)) {
                DebugOut.print("Location: " + location.holeId + "["
                        + location.numHoleExec + "] = "
                        + constantValues.get(location));
            }
        }
    }

    public void findConstantElements() {
        HashMap<AngelicCallLoc, Boolean> constantCalls = new HashMap<AngelicCallLoc, Boolean>();
        HashMap<AngelicCallLoc, Integer> constantValues = new HashMap<AngelicCallLoc, Integer>();

        // go through every trace
        for (ExecutionTrace trace : traces) {
            // go through every angelic call
            for (AngelicCallInfo angelicCall : trace.angelicCalls) {
                AngelicCallLoc location = angelicCall.location;
                // if the location is already visited and is said to be constant
                if (constantCalls.keySet().contains(location)
                        && constantCalls.get(location)) {
                    // check if location is still constant
                    if (!constantValues.get(location).equals(
                            angelicCall.valueChosen)) {
                        constantCalls.put(location, false);
                    }
                } else {
                    // check if first time visiting location
                    if (!constantCalls.keySet().contains(location)) {
                        constantCalls.put(location, true);
                        constantValues.put(location, angelicCall.valueChosen);
                    }
                }

            }
        }

        List<AngelicCallLoc> locations = new ArrayList<AngelicCallLoc>(
                constantValues.keySet());
        Collections.sort(locations);
        DebugOut.print("Constant elements");
        for (AngelicCallLoc location : locations) {
            if (constantCalls.get(location)) {
                DebugOut.print("Location: " + location.holeId + "["
                        + location.numHoleExec + "] = "
                        + constantValues.get(location));
            }
        }
    }
}
