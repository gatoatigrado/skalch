package sketch.ui;

import java.lang.reflect.Array;
import java.util.Arrays;

import javax.swing.DefaultListModel;
import javax.swing.JList;

/**
 * generic wrapper for a list view
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiList<T> {
    private static final long serialVersionUID = -21488839374096350L;
    protected DefaultListModel list_model = new DefaultListModel();
    protected JList list;
    protected Class<T[]> type_param_array_type;

    public ScUiList(JList list) {
        this.list = list;
        list.setModel(list_model);
    }

    /** FIXME - this is incredibly annoying */
    @SuppressWarnings("unchecked")
    public void add(T element) {
        type_param_array_type =
                (Class<T[]>) Array.newInstance(element.getClass(), 0)
                        .getClass();
        list_model.addElement(element);
    }

    public T[] getSelected() {
        Object[] as_obj = list.getSelectedValues();
        return Arrays.copyOf(as_obj, as_obj.length, type_param_array_type);
    }
}
