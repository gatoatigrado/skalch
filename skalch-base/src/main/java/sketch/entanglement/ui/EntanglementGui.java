package sketch.entanglement.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JTabbedPane;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.TraceSubset;
import sketch.entanglement.ui.program.ProgramDisplay;
import sketch.ui.sourcecode.ScSourceConstruct;

public class EntanglementGui extends EntanglementGuiBase implements ActionListener {

    private static final long serialVersionUID = 1L;
    
    private JTabbedPane tabbedPane;
    private JButton removeTabButton;
    
    private ProgramDisplay programDisplay;

    public EntanglementGui(List<TraceSubset> subsets, ProgramDisplay programDisplay)
    {
        super();
        this.programDisplay = programDisplay;
        tabbedPane = getTabbedPane();

        removeTabButton = getRemoveTabButton();
        removeTabButton.setActionCommand("removeTab");
        removeTabButton.addActionListener(this);

        for (TraceSubset subset : subsets) {
            addTraceSet(subset);
        }

        pack();
    }

    private void addTraceSet(TraceSubset subset) {
//        HashMap<Trace, ScStack> filteredTraceToStack = new HashMap<Trace, ScStack>();
        Set<Trace> traces = new HashSet<Trace>(subset.getTraces());

//        for (Trace trace : traces) {
//            filteredTraceToStack.put(trace, traceToStack.get(trace));
//        }

        EntanglementGuiPanel panel =
                new EntanglementGuiPanel(this, traces, programDisplay);
        tabbedPane.addTab(subset.getPartitionName(), panel);
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand() == "removeTab") {
            int tab = tabbedPane.getSelectedIndex();
            if (tab != -1) {
                tabbedPane.remove(tab);
            }
        }
    }
}
