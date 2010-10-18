package sketch.entanglement.ui;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class EntanglementGui extends EntanglementGuiBase {
    private static final long serialVersionUID = 1L;
    private SATEntanglementAnalysis satEA;
    private SimpleEntanglementAnalysis ea;
    private Set<Trace> traces;
    private List<DynAngel> angelOrder;

    public EntanglementGui(Set<Trace> traces, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        super();
        this.traces = traces;
        this.satEA = satEA;
        this.ea = ea;

        if (traces.isEmpty()) {
            return;
        }

        Trace t = traces.iterator().next();
        angelOrder = new ArrayList<DynAngel>();
        for (Event e : t.getEvents()) {
            angelOrder.add(e.dynAngel);
        }

        addSummary();
        addPartitions();
    }

    private void addSummary() {
        PartitionSummaryPanel panel = new PartitionSummaryPanel(angelOrder, satEA, ea);
        JScrollPane entanglementPane = getEntanglementPane();
        entanglementPane.getViewport().add(panel);
    }

    public void addPartitions() {
        JScrollPane partitionPane = getPartitionPane();
        JPanel partitionPanePanel = new JPanel();
        partitionPanePanel.setLayout(new BoxLayout(partitionPanePanel, BoxLayout.Y_AXIS));

        partitionPane.getViewport().add(partitionPanePanel);

        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();
        HashMap<DynAngel, Set<DynAngel>> angelToPartition =
                new HashMap<DynAngel, Set<DynAngel>>();

        for (Set<DynAngel> partition : partitions) {
            for (DynAngel angel : partition) {
                angelToPartition.put(angel, partition);
            }
        }

        for (DynAngel angel : angelOrder) {
            Set<DynAngel> partition = angelToPartition.get(angel);
            if (partitions.contains(partition)) {
                partitions.remove(partition);

                PartitionPanel panel = new PartitionPanel(partition, ea);
                partitionPanePanel.add(panel);
            }

        }

        // partitionPanel.add(checkBox);
    }
}
