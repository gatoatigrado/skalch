package sketch.entanglement.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class PartitionSummaryPanel extends JPanel {
    private Set<Set<DynAngel>> partitions;
    private SATEntanglementAnalysis satEA;
    private SimpleEntanglementAnalysis ea;
    private Map<Trace, ScStack> traceToStack;
    private Color[][] color;

    public PartitionSummaryPanel(Map<Trace, ScStack> traceToStack,
            List<DynAngel> angelOrder, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        this.traceToStack = traceToStack;
        this.satEA = satEA;
        this.ea = ea;
        color = new EntanglementColoring(ea, satEA).getColorMatrix();

        partitions = satEA.getEntangledPartitions();
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        HashMap<DynAngel, Integer> colorMapping = new HashMap<DynAngel, Integer>();

        Random random = new Random();

        for (DynAngel angel : angelOrder) {
            JLabel angelLabel = new JLabel(angel.toString(), SwingConstants.CENTER);
            angelLabel.setOpaque(true);
            angelLabel.setMaximumSize(new Dimension(Short.MAX_VALUE, Short.MAX_VALUE));

            Color c = color[angel.staticAngelId][angel.execNum];
            if (c == null) {
                c = Color.white;
            }
            angelLabel.setBackground(c);
            add(angelLabel);
        }
        setVisible(true);

    }

    public Map<Trace, ScStack> getTraceToStack() {
        return traceToStack;
    }

    public SATEntanglementAnalysis getSatEA() {
        return satEA;
    }

    public SimpleEntanglementAnalysis getEA() {
        return ea;
    }
}
