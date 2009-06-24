package sketch.ui.gui;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.event.ListSelectionEvent;

import sketch.dyn.ctrls.ScCtrlSourceInfo;
import sketch.dyn.inputs.ScCounterexample;
import sketch.dyn.synth.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.sourcecode.ScSourceCache;
import sketch.ui.sourcecode.ScSourceLocation;
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
        HashMap<String, Vector<ScCtrlSourceInfo>> info_by_filename =
                new HashMap<String, Vector<ScCtrlSourceInfo>>();
        for (ScCtrlSourceInfo hole_info : ui_thread.sketch.ctrl_src_info) {
            String f = hole_info.src_loc.filename;
            if (!info_by_filename.containsKey(f)) {
                info_by_filename.put(f, new Vector<ScCtrlSourceInfo>());
            }
            info_by_filename.get(f).add(hole_info);
        }
        stack.set_fixed_for_illustration(ui_thread.sketch);
        ScSourceCache.singleton().add_filenames(info_by_filename.keySet());
        StringBuilder result = new StringBuilder();
        result.append("<html>\n  <head>\n<style>\nbody {\n"
                + "font-size: 12pt;\n}\n</style>\n  </head>\n  <body>"
                + "<p>color indicates how often values are changed: red "
                + "is very often, yellow is occasionally, blue is never.");
        for (Entry<String, Vector<ScCtrlSourceInfo>> entry : info_by_filename
                .entrySet())
        {
            result.append("\n<p><pre style=\"font-family: serif;\">");
            add_source_info(result, entry.getKey(), entry.getValue());
            result.append("\n</pre></p>");
        }
        result.append("\n</body>\n</html>");
        sourceCodeEditor.setText(result.toString());
    }

    private void add_source_info(StringBuilder result, String key,
            Vector<ScCtrlSourceInfo> value)
    {
        ScCtrlSourceInfo[] hole_info_sorted =
                value.toArray(new ScCtrlSourceInfo[0]);
        Arrays.sort(hole_info_sorted);
        ScSourceLocation start = hole_info_sorted[0].src_loc.contextBefore(3);
        ScSourceLocation end =
                hole_info_sorted[hole_info_sorted.length - 1].src_loc
                        .contextAfter(3);
        ScStackSourceVisitor v = new ScStackSourceVisitor();
        // starting context
        result.append(v.visitCode(start));
        // visit constructs and all code in between
        for (int a = 0; a < hole_info_sorted.length; a++) {
            result.append(v.visitHoleInfo(hole_info_sorted[a]));
            if (a + 1 < hole_info_sorted.length) {
                result.append(v.visitCode(hole_info_sorted[a].src_loc
                        .source_between(hole_info_sorted[a + 1].src_loc)));
            }
        }
        result.append(v.visitCode(end));
    }
}
