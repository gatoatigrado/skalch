package sketch.entanglement.ui;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.ui.sourcecode.ScSourceConstruct;

public class EntanglementSummaryGui extends EntanglementSummaryGuiBase implements
        MouseListener
{

    private JScrollPane subsetScrollPane;
    private JPanel scrollPanel;
    private ScDynamicSketchCall<?> sketch;
    private Set<ScSourceConstruct> sourceCodeInfo;

    public EntanglementSummaryGui(ScDynamicSketchCall<?> sketch,
            Set<ScSourceConstruct> sourceCodeInfo)
    {
        this.sketch = sketch;
        this.sourceCodeInfo = sourceCodeInfo;

        subsetScrollPane = getSubsetScrollPane();
        scrollPanel = new JPanel();
        subsetScrollPane.getViewport().add(scrollPanel);
        scrollPanel.setLayout(new BoxLayout(scrollPanel, BoxLayout.X_AXIS));
    }

    public void addTraceSubset(List<DynAngel> angelOrder,
            HashMap<Trace, ScStack> subsetTraceToStack)
    {
        SATEntanglementAnalysis satEA =
                new SATEntanglementAnalysis(subsetTraceToStack.keySet());
        SimpleEntanglementAnalysis ea =
                new SimpleEntanglementAnalysis(subsetTraceToStack.keySet());

        PartitionSummaryPanel panel =
                new PartitionSummaryPanel(subsetTraceToStack, angelOrder, satEA, ea);
        panel.addMouseListener(this);
        scrollPanel.add(panel);
    }

    public void mouseClicked(MouseEvent event) {
    // PartitionSummaryPanel panel = (PartitionSummaryPanel) event.getSource();
    // EntanglementGui subsetGui =
    // new EntanglementGui(panel.getTraceToStack(), panel.getSatEA(),
    // panel.getEA(), sketch, sourceCodeInfo);
    // subsetGui.setVisible(true);
    }

    public void mouseEntered(MouseEvent arg0) {
    // TODO Auto-generated method stub

    }

    public void mouseExited(MouseEvent arg0) {
    // TODO Auto-generated method stub

    }

    public void mousePressed(MouseEvent arg0) {
    // TODO Auto-generated method stub

    }

    public void mouseReleased(MouseEvent arg0) {
    // TODO Auto-generated method stub

    }
}
