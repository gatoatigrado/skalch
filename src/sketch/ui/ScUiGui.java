package sketch.ui;

import javax.swing.event.ListSelectionEvent;

import sketch.dyn.inputs.ScCounterexample;
import sketch.ui.modifiers.ScModifierDispatcher;

/**
 * kinda because I can't resist using Eclipse's formatter (it also separates
 * generated code). This file is unfortunately long, but it seems nicer with
 * inner classes.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiGui extends gui_0_1 {
    private static final long serialVersionUID = 6584583626375432139L;
    public ScUiThread ui_thread;
    public ScUiList<ScCounterexample> inputChoices;
    public ScUiList<ScModifierDispatcher> synthCompletions;

    @SuppressWarnings("unchecked")
    /** FIXME - this is incredibly annoying */
    public ScUiGui(ScUiThread ui_thread) {
        super();
        this.ui_thread = ui_thread;
        // java is very annoying
        inputChoices =
                new ScUiList<ScCounterexample>(selectInputList,
                        (Class<ScCounterexample[]>) (new ScCounterexample[0])
                                .getClass());
        synthCompletions =
                new ScUiList<ScModifierDispatcher>(
                        synthCompletionList,
                        (Class<ScModifierDispatcher[]>) (new ScModifierDispatcher[0])
                                .getClass());
    }

    // === ui functions ===

    @Override
    protected void synthCompletionSelectionChanged(ListSelectionEvent evt) {
        ScModifierDispatcher[] selected = synthCompletions.getSelected();
        viewSelectionsButton.setEnabled(selected.length == 1);
    }

    @Override
    protected void viewSelections() {
        synthCompletions.getSelected()[0].dispatch();
    }
}
