package sketch.entanglement.ui;

import java.util.Vector;

import javax.swing.DefaultListModel;

public class PartitionListModel extends DefaultListModel {

    public PartitionListModel(Vector<? extends Object> list) {
        for (Object element : list) {
            addElement(element);
        }
    }

    public void updateIndex(int index) {
        fireContentsChanged(this, index, index);
    }
}
