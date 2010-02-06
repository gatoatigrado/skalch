package sketch.ui.entanglement;

import java.util.ArrayList;
import java.util.List;

public class ExecutionTrace {
    List<AngelicCallInfo> angelicCalls;

    public ExecutionTrace() {
        angelicCalls = new ArrayList<AngelicCallInfo>();
    }

    public void addAngelicCall(int holeId, int numHoleExec, int numValues,
            int valueChosen) {
        angelicCalls.add(new AngelicCallInfo(holeId, numHoleExec, numValues,
                valueChosen));
    }
}
