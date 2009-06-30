package sketch.ui.gui;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.event.ListSelectionEvent;

import sketch.dyn.BackendOptions;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.synth.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.sourcecode.ScSourceCache;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceLocation;
import sketch.ui.sourcecode.ScStackSourceVisitor;
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
    public ScUiList<ScFixedInputConf> inputChoices;
    public ScUiList<ScModifierDispatcher> synthCompletions;
    public int num_synth_active = 0;
    public int context_len;
    public int context_split_len;

    @SuppressWarnings("unchecked")
    /** FIXME - this is incredibly annoying */
    public ScUiGui(ScUiThread ui_thread) {
        super();
        this.ui_thread = ui_thread;
        context_len = (int) BackendOptions.ui_opts.long_("context_len");
        context_split_len =
                (int) BackendOptions.ui_opts.long_("context_split_len");
        // java is very annoying
        inputChoices =
                new ScUiList<ScFixedInputConf>(selectInputList,
                        (Class<ScFixedInputConf[]>) (new ScFixedInputConf[0])
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
        HashMap<String, Vector<ScSourceConstruct>> info_by_filename =
                new HashMap<String, Vector<ScSourceConstruct>>();
        for (ScSourceConstruct hole_info : ui_thread.sketch.ctrl_src_info) {
            String f = hole_info.entire_location.filename;
            if (!info_by_filename.containsKey(f)) {
                info_by_filename.put(f, new Vector<ScSourceConstruct>());
            }
            info_by_filename.get(f).add(hole_info);
        }
        stack.set_fixed_for_illustration(ui_thread.sketch);
        ScSourceCache.singleton().add_filenames(info_by_filename.keySet());
        StringBuilder result = new StringBuilder();
        result.append("<html>\n  <head>\n<style>\nbody {\n"
                + "font-size: 12pt;\n}\n</style>\n  </head>\n  "
                + "<body style=\"margin-top: 0px;\">"
                + "<p style=\"margin-top: 0.1em;\">"
                + "color indicates how often values are changed: red "
                + "is very often, yellow is occasionally, blue is never.");
        for (Entry<String, Vector<ScSourceConstruct>> entry : info_by_filename
                .entrySet())
        {
            result.append("\n<p><pre style=\"font-family: serif;\">");
            add_source_info(result, entry.getKey(), entry.getValue());
            result.append("</pre></p><hr />");
        }
        result.append("<p style=\"color: #aaaaaa\">Stack view (in case "
                + "there are bugs above or it's less readable)<br />\n");
        result.append(stack.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");
        sourceCodeEditor.setText(result.toString());
    }

    private void add_source_info(StringBuilder result, String key,
            Vector<ScSourceConstruct> vector)
    {
        ScSourceConstruct[] hole_info_sorted =
                vector.toArray(new ScSourceConstruct[0]);
        Arrays.sort(hole_info_sorted);
        ScSourceLocation start =
                hole_info_sorted[0].entire_location.contextBefore(context_len);
        ScSourceLocation end =
                hole_info_sorted[hole_info_sorted.length - 1].entire_location
                        .contextAfter(context_len);
        ScStackSourceVisitor v = new ScStackSourceVisitor();
        // starting context
        result.append(v.visitCode(start));
        // visit constructs and all code in between
        for (int a = 0; a < hole_info_sorted.length; a++) {
            result.append(v.visitHoleInfo(hole_info_sorted[a]));
            ScSourceLocation loc = hole_info_sorted[a].entire_location;
            if (a + 1 < hole_info_sorted.length) {
                ScSourceLocation next_loc =
                        hole_info_sorted[a + 1].entire_location;
                ScSourceLocation between = loc.source_between(next_loc);
                if (between.numLines() >= context_split_len) {
                    // split the context
                    result.append(v.visitCode(loc.contextAfter(context_len)));
                    result.append("</pre>\n<hr /><pre>");
                    result.append(v.visitCode(next_loc
                            .contextBefore(context_len)));
                } else {
                    result.append(v.visitCode(loc.source_between(next_loc)));
                }
            }
        }
        result.append(v.visitCode(end));
    }

    public void disableStopButton() {
        stopButton.setEnabled(false);
    }
}
