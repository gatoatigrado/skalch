package sketch.entanglement.ui;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.ListSelectionModel;

import sketch.entanglement.DynAngel;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;

public class PartitionPanel extends JPanel implements ActionListener {

    private enum Marking {
        GOOD, UNKNOWN, BAD
    }

    class ValueListElement {

        private final Trace trace;
        private Marking marking;

        public ValueListElement(Trace trace) {
            this.trace = trace;
            marking = Marking.UNKNOWN;
        }

        public Trace getTrace() {
            return trace;
        }

        @Override
        public String toString() {
            String returnString = "";
            switch (marking) {
                case GOOD:
                    returnString += "+";
                    break;
                case BAD:
                    returnString += "-";
                    break;
                default:

            }
            returnString += trace.toString();
            return returnString;
        }

        public void mark(Marking m) {
            marking = m;
            System.out.println(marking);
        }
    }

    class PopupListener extends MouseAdapter {
        final private JPopupMenu popup;
        final private JList list;

        PopupListener(JList list, JPopupMenu popup) {
            this.list = list;
            this.popup = popup;
        }

        @Override
        public void mousePressed(MouseEvent e) {
            maybeShowPopup(e);
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            maybeShowPopup(e);
        }

        private void maybeShowPopup(MouseEvent e) {
            if (e.isPopupTrigger()) {
                int index = list.locationToIndex(e.getPoint());
                list.setSelectedIndex(index);
                popup.show(e.getComponent(), e.getX(), e.getY());
            }
        }
    }

    private List<DynAngel> angels;
    private SimpleEntanglementAnalysis ea;
    private Set<Trace> values;

    private Set<Trace> goodValues;
    private Set<Trace> badValues;

    private JList list;
    private JLabel label;
    private PartitionListModel model;
    private Vector<ValueListElement> listValues;

    public PartitionPanel(Set<DynAngel> partition, SimpleEntanglementAnalysis ea) {
        angels = new ArrayList<DynAngel>(partition);
        Collections.sort(angels);
        this.ea = ea;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        addLabel(this);

        values = ea.getValues(partition);
        addList(this);
        addPopupMenu();
    }

    private void addLabel(JPanel panel) {
        String text = "Choose: ";
        for (int i = 0; i < angels.size(); i++) {
            text += angels.get(i);
            if (i != angels.size() - 1) {
                text += ",";
            }
        }
        label = new JLabel(text);
        label.setAlignmentX(JLabel.LEFT_ALIGNMENT);
        panel.add(label);
    }

    private void addList(JPanel panel) {
        listValues = new Vector<ValueListElement>();

        for (Trace value : values) {
            listValues.add(new ValueListElement(value));
        }

        label.setText("[" + listValues.size() + "]" + label.getText());

        model = new PartitionListModel(listValues);
        list = new JList(model);
        list.setSelectedIndex(0);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setMaximumSize(new Dimension(Short.MAX_VALUE, Short.MAX_VALUE));
        list.setAlignmentX(JLabel.LEFT_ALIGNMENT);
        panel.add(list);
    }

    private void addPopupMenu() {
        JMenuItem menuItem;

        // Create the popup menu.
        JPopupMenu popup = new JPopupMenu();
        menuItem = new JMenuItem("Mark as good");
        menuItem.setActionCommand("good");
        menuItem.addActionListener(this);
        popup.add(menuItem);
        menuItem = new JMenuItem("Mark as bad");
        menuItem.setActionCommand("bad");
        menuItem.addActionListener(this);
        popup.add(menuItem);
        menuItem = new JMenuItem("Reset");
        menuItem.setActionCommand("reset");
        menuItem.addActionListener(this);
        popup.add(menuItem);

        // Add listener to the text area so the popup menu can come up.
        MouseListener popupListener = new PopupListener(list, popup);
        list.addMouseListener(popupListener);
    }

    public Trace getSelectedValue() {
        Object values[] = list.getSelectedValues();
        if (values.length == 1) {
            return ((ValueListElement) values[0]).getTrace();
        }
        return null;
    }

    public JList getList() {
        return list;
    }

    public void actionPerformed(ActionEvent event) {
        ValueListElement selected = (ValueListElement) list.getSelectedValue();
        JMenuItem item = (JMenuItem) event.getSource();
        if ("good".equals(item.getActionCommand())) {
            selected.mark(Marking.GOOD);
        } else if ("bad".equals(item.getActionCommand())) {
            selected.mark(Marking.BAD);
        } else if ("reset".equals(item.getActionCommand())) {
            selected.mark(Marking.UNKNOWN);
        }
        int index = list.getSelectedIndex();
        model.updateIndex(index);
    }

    public Set<Trace> getGoodTraces() {
        HashSet<Trace> goodTraces = new HashSet<Trace>();
        for (ValueListElement listValue : listValues) {
            if (listValue.marking == Marking.GOOD) {
                goodTraces.add(listValue.trace);
            }
        }
        return goodTraces;
    }

    public Set<Trace> getBadTraces() {
        HashSet<Trace> badTraces = new HashSet<Trace>();
        for (ValueListElement listValue : listValues) {
            if (listValue.marking == Marking.BAD) {
                badTraces.add(listValue.trace);
            }
        }
        return badTraces;
    }

    public Set<DynAngel> getAngelPartition() {
        return new HashSet<DynAngel>(angels);
    }
}
