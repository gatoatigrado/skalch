package sketch.entanglement.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JList;

import sketch.entanglement.DynAngel;

public class DisentanglementGui extends DisentanglementGuiBase implements ActionListener {

    private Vector<DynAngel> angelOrder;
    private Vector<DynAngel> originalAngelOrder;

    private Vector<Set<DynAngel>> partitions;

    private JList chooseList;
    private JList partitionList;
    private EntanglementGuiPanel parent;

    public DisentanglementGui(EntanglementGuiPanel parent, boolean modal,
            List<DynAngel> angelOrder)
    {
        super(parent.getFrame(), modal);
        partitions = new Vector<Set<DynAngel>>();
        originalAngelOrder = new Vector<DynAngel>(angelOrder);
        this.angelOrder = new Vector<DynAngel>(angelOrder);
        this.parent = parent;

        chooseList = getChooseList();
        chooseList.setListData(this.angelOrder);

        partitionList = getPartitionList();

        JButton createButton = getCreateButton();
        createButton.setActionCommand("create");
        createButton.addActionListener(this);

        JButton removeButton = getRemoveButton();
        removeButton.setActionCommand("remove");
        removeButton.addActionListener(this);

        JButton refineButton = getRefineButton();
        refineButton.setActionCommand("refine");
        refineButton.addActionListener(this);
    }

    public void actionPerformed(ActionEvent event) {
        if ("create".equals(event.getActionCommand())) {
            Object[] selected = chooseList.getSelectedValues();
            if (selected.length > 0) {
                HashSet<DynAngel> partition = new HashSet<DynAngel>();
                for (Object angel : selected) {
                    partition.add((DynAngel) angel);
                }
                angelOrder.removeAll(partition);
                partitions.add(partition);

                chooseList.setListData(angelOrder);
                partitionList.setListData(partitions);
            }
        } else if ("remove".equals(event.getActionCommand())) {
            Object[] selected = partitionList.getSelectedValues();
            if (selected.length > 0) {
                HashSet<DynAngel> addedAngels = new HashSet<DynAngel>();
                for (Object partition : selected) {
                    HashSet<DynAngel> partitionSet = (HashSet<DynAngel>) partition;
                    addedAngels.addAll(partitionSet);
                    partitions.remove(partitionSet);
                }
                Vector<DynAngel> newAngelOrder = new Vector<DynAngel>();
                for (DynAngel angel : originalAngelOrder) {
                    if (angelOrder.contains(angel) || addedAngels.contains(angel)) {
                        newAngelOrder.add(angel);
                    }
                }

                angelOrder = newAngelOrder;
                chooseList.setListData(angelOrder);
                partitionList.setListData(partitions);
            }
        } else if ("refine".equals(event.getActionCommand())) {
            parent.partitionTraces(new HashSet<Set<DynAngel>>(partitions));
            setVisible(false);
            dispose();
        }
    }
}
