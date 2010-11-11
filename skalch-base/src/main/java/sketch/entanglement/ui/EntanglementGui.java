package sketch.entanglement.ui;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JTabbedPane;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.SubsetOfTraces;
import sketch.ui.sourcecode.ScSourceConstruct;

public class EntanglementGui extends EntanglementGuiBase {

    private ScDynamicSketchCall<?> sketch;
    private Set<ScSourceConstruct> sourceCodeInfo;
    private JTabbedPane tabbedPane;
    private Map<Trace, ScStack> traceToStack;

    public EntanglementGui(List<SubsetOfTraces> subsets,
            Map<Trace, ScStack> traceToStack, ScDynamicSketchCall<?> sketch,
            Set<ScSourceConstruct> sourceCodeInfo)
    {
        super();
        this.sketch = sketch;
        this.sourceCodeInfo = sourceCodeInfo;
        this.traceToStack = traceToStack;
        tabbedPane = getTabbedPane();

        for (SubsetOfTraces subset : subsets) {
            addTraceSet(subset);
        }

        pack();
    }

    private void addTraceSet(SubsetOfTraces subset) {
        HashMap<Trace, ScStack> filteredTraceToStack = new HashMap<Trace, ScStack>();
        Set<Trace> traces = new HashSet<Trace>(subset.getTraces());

        for (Trace trace : traces) {
            filteredTraceToStack.put(trace, traceToStack.get(trace));
        }

        EntanglementGuiPanel panel =
                new EntanglementGuiPanel(this, filteredTraceToStack, sketch,
                        sourceCodeInfo);
        tabbedPane.addTab(subset.getPartitionName(), panel);
    }
}
