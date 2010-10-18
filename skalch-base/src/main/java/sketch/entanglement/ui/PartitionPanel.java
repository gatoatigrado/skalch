package sketch.entanglement.ui;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;

import sketch.entanglement.DynAngel;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;

public class PartitionPanel extends JPanel {

    private List<DynAngel> angels;
    private SimpleEntanglementAnalysis ea;
    private Set<Trace> values;

    public PartitionPanel(Set<DynAngel> partition, SimpleEntanglementAnalysis ea) {
        angels = new ArrayList<DynAngel>(partition);
        Collections.sort(angels);
        this.ea = ea;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        addLabel();

        values = ea.getValues(partition);
        addLists();
    }

    private void addLabel() {
        String label = "";
        for (int i = 0; i < angels.size(); i++) {
            label += angels.get(i);
            if (i != angels.size() - 1) {
                label += ",";
            }
        }
        System.out.println(label);
        add(new JLabel(label));
    }

    private void addLists() {
        add(new JList(values.toArray()));
    }

}
