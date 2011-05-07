package sketch.entanglement.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.util.List;
import java.util.Map;
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
    private static final long serialVersionUID = 1L;
    
    public PartitionSummaryPanel(List<DynAngel> angelOrder, EntanglementColoring ec)
    {
        Color[][] color = ec.getColorMatrix();

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

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
}
