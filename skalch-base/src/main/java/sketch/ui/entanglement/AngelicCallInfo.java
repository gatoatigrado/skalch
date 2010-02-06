package sketch.ui.entanglement;

public class AngelicCallInfo {

    final AngelicCallLoc location;
    final int numValues;
    final int valueChosen;

    public AngelicCallInfo(int holeId, int numHoleExec, int numValues,
            int valueChosen) {
        location = new AngelicCallLoc(holeId, numHoleExec);
        this.numValues = numValues;
        this.valueChosen = valueChosen;
    }
}
