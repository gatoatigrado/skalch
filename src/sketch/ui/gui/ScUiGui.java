package sketch.ui.gui;

import java.util.Arrays;
import java.util.HashSet;

import javax.swing.event.ListSelectionEvent;

import sketch.dyn.ctrls.ScCtrlSourceInfo;
import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.synth.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.util.DebugOut;

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
        acceptButton.setEnabled(selected.length == 1
                && selected[0].isAcceptable());
    }

    @Override
    protected void viewSelections() {
        synthCompletions.getSelected()[0].dispatch();
    }

    @Override
    protected void acceptSolution() {
        DebugOut.not_implemented("accept solutions");
    }

    @Override
    protected void hyperlinkClicked(String linkurl) {
    }

    @Override
    protected void setMonospace(boolean monospace) {
    }

    @Override
    protected void stopSolver() {
        ui_thread.ssr.wait_handler.set_synthesis_complete();
    }

    public void fillWithStack(ScStack stack) {
        // get source
        HashSet<String> filenames = new HashSet<String>();
        for (ScCtrlSourceInfo hole_info : ui_thread.sketch.ctrl_src_info) {
            filenames.add(hole_info.src_loc.filename);
        }
        DebugOut.print_mt("hole filenames", filenames);
        ScCtrlSourceInfo[] hole_info_sorted =
                ui_thread.sketch.ctrl_src_info.toArray(new ScCtrlSourceInfo[0]);
        Arrays.sort(hole_info_sorted);
        for (ScCtrlSourceInfo hole_info : hole_info_sorted) {
            DebugOut.print_mt(hole_info);
        }
    }
}
