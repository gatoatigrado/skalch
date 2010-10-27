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
    private HashMap<Integer, Color> colorTable;
    private SATEntanglementAnalysis satEA;
    private SimpleEntanglementAnalysis ea;
    private Map<Trace, ScStack> traceToStack;

    public PartitionSummaryPanel(Map<Trace, ScStack> traceToStack,
            List<DynAngel> angelOrder, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        this.traceToStack = traceToStack;
        this.satEA = satEA;
        this.ea = ea;

        partitions = satEA.getEntangledPartitions();
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        HashMap<DynAngel, Integer> colorMapping = new HashMap<DynAngel, Integer>();

        int i = 0;

        for (Set<DynAngel> partition : partitions) {
            if (partition.size() > 1) {
                i++;
                for (DynAngel angel : partition) {
                    colorMapping.put(angel, i);
                }
            } else {
                colorMapping.put(partition.iterator().next(), 0);
            }
        }

        Random random = new Random();
        colorTable = new HashMap<Integer, Color>();
        colorTable.put(0, Color.WHITE);

        for (DynAngel angel : angelOrder) {
            JLabel angelLabel = new JLabel(angel.toString(), SwingConstants.CENTER);
            angelLabel.setOpaque(true);
            angelLabel.setBackground(getColor(colorMapping.get(angel), random));
            angelLabel.setMaximumSize(new Dimension(Short.MAX_VALUE, Short.MAX_VALUE));
            add(angelLabel);
        }
        setVisible(true);

    }

    private Color getColor(Integer i, Random random) {
        if (colorTable.containsKey(i)) {
            return colorTable.get(i);
        } else {
            Color color = Color.WHITE;
            while (colorTable.containsValue(color)) {
                color =
                        new Color(random.nextInt(255), random.nextInt(255),
                                random.nextInt(255));
            }
            colorTable.put(i, color);
            return color;
        }
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
