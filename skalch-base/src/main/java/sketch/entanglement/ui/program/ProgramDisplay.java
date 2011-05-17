package sketch.entanglement.ui.program;

import sketch.entanglement.Trace;
import sketch.entanglement.ui.EntanglementColoring;

public interface ProgramDisplay {

    class ProgramOutput {
        public ProgramOutput(String text, String output) {
            this.programText = text;
            this.debugOutput = output;
        }

        public String programText;
        public String debugOutput;
    }

    //public ProgramOutput getProgramOutput(Trace selectedTrace, EntanglementColoring color);

    public String getDebugOutput(Trace selectedTrace, EntanglementColoring color);

    public String getProgramText(Trace selectedTrace, EntanglementColoring color);
}
