package sketch.ui.gui;

import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.JComponent;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.event.ListSelectionEvent;

import sketch.dyn.BackendOptions;
import sketch.dyn.debug.ScDebugEntry;
import sketch.dyn.debug.ScDebugSketchRun;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiSortedList;
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
    public ScUiSortedList<ScModifierDispatcher> synthCompletions;
    public int num_synth_active = 0;
    public int context_len;
    public int context_split_len;

    @SuppressWarnings("unchecked")
    /** FIXME - this is incredibly annoying */
    public ScUiGui(ScUiThread ui_thread) {
        super();
        this.ui_thread = ui_thread;
        KeyStroke escKeyStroke =
                KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0, false);
        Action escapeAction = new AbstractAction() {
            private static final long serialVersionUID = 1173509525674124142L;

            public void actionPerformed(ActionEvent e) {
                stopSolver();
                setVisible(false);
                dispose();
                ScUiGui.this.ui_thread.set_stop();
            }
        };
        getRootPane().getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(
                escKeyStroke, "ESCAPE");
        getRootPane().getActionMap().put("ESCAPE", escapeAction);
        context_len = (int) BackendOptions.ui_opts.long_("context_len");
        context_split_len =
                (int) BackendOptions.ui_opts.long_("context_split_len");
        // java is very annoying
        int max_list_length =
                (int) BackendOptions.ui_opts.long_("max_list_length");
        inputChoices =
                new ScUiList<ScFixedInputConf>(selectInputList,
                        (Class<ScFixedInputConf[]>) (new ScFixedInputConf[0])
                                .getClass(), max_list_length);
        synthCompletions =
                new ScUiSortedList<ScModifierDispatcher>(
                        synthCompletionList,
                        (Class<ScModifierDispatcher[]>) (new ScModifierDispatcher[0])
                                .getClass(), max_list_length);
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
        DebugOut.todo("set monospace to", monospace,
                "doesn't work with jdk 1.7");
    }

    @Override
    protected void stopSolver() {
        ui_thread.synth_runtime.wait_handler.set_synthesis_complete();
    }

    /** this all happens on the UI thread, but it shouldn't be that slow */
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
        add_debug_info(stack);
        if (!BackendOptions.ui_opts.bool_("no_scroll_topleft")) {
            scroll_topleft();
        }
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

    private void scroll_topleft() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                debugOutPane.getHorizontalScrollBar().setValue(0);
                debugOutPane.getVerticalScrollBar().setValue(0);
                sourceCodePane.getHorizontalScrollBar().setValue(0);
                sourceCodePane.getVerticalScrollBar().setValue(0);
            }
        });
    }

    /**
     * reruns the stack, collecting any debug print statements. NOTE - keep this
     * in sync with ScLocalStackSynthesis
     */
    private void add_debug_info(ScStack stack) {
        ui_thread.sketch.enable_debug();
        stack.set_for_synthesis(ui_thread.sketch);
        ScDebugSketchRun sketch_run =
                new ScDebugSketchRun(ui_thread.sketch, stack,
                        ui_thread.all_counterexamples);
        sketch_run.run();
        //
        StringBuilder debug_text = new StringBuilder();
        debug_text.append("<html>\n  <head>\n<style>\n"
                + "body {\nfont-size: 12pt;\n}\n"
                + "ul {\nmargin-left: 20pt;\n}\n</style>\n  </head>"
                + "\n  <body>\n<ul>");
        LinkedList<String> html_contexts = new LinkedList<String>();
        html_contexts.add("body");
        html_contexts.add("ul");
        for (ScDebugEntry debug_entry : sketch_run.debug_out) {
            debug_text.append(debug_entry.htmlString(html_contexts));
        }
        debug_text.append("\n</ul>\n");
        if (sketch_run.assert_failed()) {
            StackTraceElement assert_info = sketch_run.assert_info;
            debug_text.append(String.format("<p>failure at %s (line %d)</p>",
                    assert_info.getMethodName(), assert_info.getLineNumber()));
        } else {
            debug_text.append("<p>dysketch_main returned false</p>");
        }
        debug_text.append("  </body>\n</html>\n");
        debugOutEditor.setText(debug_text.toString());
    }

    public void disableStopButton() {
        stopButton.setEnabled(false);
    }
}
