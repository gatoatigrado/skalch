package sketch.ui.gui;

import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.AbstractAction;
import javax.swing.JComponent;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.event.ListSelectionEvent;

import sketch.dyn.BackendOptions;
import sketch.dyn.debug.ScDebugEntry;
import sketch.dyn.debug.ScDebugGaRun;
import sketch.dyn.debug.ScDebugRun;
import sketch.dyn.debug.ScDebugStackRun;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.ScUiSortedList;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.sourcecode.ScSourceCache;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceLocation;
import sketch.ui.sourcecode.ScSourceTraceVisitor;
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
        setupKeys();
        context_len = BackendOptions.ui_opts.context_len;
        context_split_len = BackendOptions.ui_opts.context_split_len;
        // java is very annoying
        int max_list_length = BackendOptions.ui_opts.max_list_length;
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

    @SuppressWarnings("serial")
    public abstract class KeyAction extends AbstractAction {
        public abstract void action();

        public final void add(int key_event0, boolean key_release) {
            KeyStroke my_keystroke =
                    KeyStroke.getKeyStroke(key_event0, 0, key_release);
            getRootPane().getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(
                    my_keystroke, this);
            getRootPane().getActionMap().put(this, this);
        }

        public final void actionPerformed(ActionEvent e) {
            action();
        }
    }

    /** setup keyboard shortcuts */
    protected void setupKeys() {
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                stopSolver();
                setVisible(false);
                dispose();
                ui_thread.set_stop();
            }
        }).add(KeyEvent.VK_ESCAPE, false);
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                ScModifierDispatcher[] selected =
                        synthCompletions.getSelected();
                if (selected.length >= 1) {
                    synthCompletions.select_next(selected[0]).dispatch();
                }
            }
        }).add(KeyEvent.VK_J, true);
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                ScModifierDispatcher[] selected =
                        synthCompletions.getSelected();
                if (selected.length >= 1) {
                    synthCompletions.select_prev(selected[0]).dispatch();
                }
            }
        }).add(KeyEvent.VK_K, true);
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
        stack.set_fixed_for_illustration(ui_thread.sketch);
        StringBuilder result = getSourceWithSynthesisValues();
        result.append("<p style=\"color: #aaaaaa\">Stack view (in case "
                + "there are bugs above or it's less readable)<br />\n");
        result.append(stack.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");
        sourceCodeEditor.setText(result.toString());
        add_debug_info(new ScDebugStackRun(ui_thread.sketch, stack,
                ui_thread.all_counterexamples));
        if (!BackendOptions.ui_opts.no_scroll_topleft) {
            scroll_topleft();
        }
    }

    /**
     * get a string builder with html representing the source and filled in
     * values; most work done by add_source_info()
     */
    protected StringBuilder getSourceWithSynthesisValues() {
        HashMap<String, Vector<ScSourceConstruct>> info_by_filename =
                new HashMap<String, Vector<ScSourceConstruct>>();
        for (ScSourceConstruct hole_info : ui_thread.sketch.construct_src_info)
        {
            String f = hole_info.entire_location.filename;
            if (!info_by_filename.containsKey(f)) {
                info_by_filename.put(f, new Vector<ScSourceConstruct>());
            }
            info_by_filename.get(f).add(hole_info);
        }
        ScSourceCache.singleton().add_filenames(info_by_filename.keySet());
        StringBuilder result = new StringBuilder();
        result.append("<html>\n  <head>\n<style>\nbody {\n"
                + "font-size: 12pt;\n}\n</style>\n  </head>\n  "
                + "<body style=\"margin-top: 0px;\">"
                + "<p style=\"margin-top: 0.1em;\">"
                + "color indicates how often values are changed: red "
                + "is very often, yellow is occasionally, blue is never.</p>");
        for (Entry<String, Vector<ScSourceConstruct>> entry : info_by_filename
                .entrySet())
        {
            result.append("\n<p><pre style=\"font-family: serif;\">");
            add_source_info(result, entry.getKey(), entry.getValue());
            result.append("</pre></p><hr />");
        }
        return result;
    }

    /** sub-method for the above (getSourceWithSynthesisValues) */
    protected void add_source_info(StringBuilder result, String key,
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
        ScSourceTraceVisitor v = new ScSourceTraceVisitor();
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
     * reruns the stack, collecting any debug print statements.
     */
    private void add_debug_info(ScDebugRun debug_run) {
        debug_run.run();
        //
        StringBuilder debug_text = new StringBuilder();
        debug_text.append("<html>\n  <head>\n<style>\n"
                + "body {\nfont-size: 12pt;\n}\n"
                + "ul {\nmargin-left: 20pt;\n}\n</style>\n  </head>"
                + "\n  <body>\n<ul>");
        LinkedList<String> html_contexts = new LinkedList<String>();
        html_contexts.add("body");
        html_contexts.add("ul");
        for (ScDebugEntry debug_entry : debug_run.debug_out) {
            debug_text.append(debug_entry.htmlString(html_contexts));
        }
        debug_text.append("\n</ul>\n");
        if (debug_run.assert_failed()) {
            StackTraceElement assert_info = debug_run.assert_info;
            debug_text.append(String.format("<p>failure at %s (line %d)</p>",
                    assert_info.getMethodName(), assert_info.getLineNumber()));
        } else {
            debug_text.append("<p>dysketch_main returned "
                    + (debug_run.succeeded ? "true" : "false") + "</p>");
        }
        debug_text.append("  </body>\n</html>\n");
        debugOutEditor.setText(debug_text.toString());
    }

    public void disableStopButton() {
        stopButton.setEnabled(false);
    }

    public void fillWithGaIndividual(ScGaIndividual individual) {
        ScGaIndividual clone = individual.clone();
        clone.reset_fitness();
        clone.set_for_synthesis_and_reset(ui_thread.sketch,
                ui_thread.ga_ctrl_conf, ui_thread.ga_oracle_conf);
        StringBuilder result = getSourceWithSynthesisValues();
        result.append("<p style=\"color: #aaaaaa;\"> ga synthesis "
                + "individual<br />\n");
        result.append(individual.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");
        sourceCodeEditor.setText(result.toString());
        add_debug_info(new ScDebugGaRun(ui_thread.sketch,
                ui_thread.all_counterexamples, individual,
                ui_thread.ga_ctrl_conf, ui_thread.ga_oracle_conf));
        if (!BackendOptions.ui_opts.no_scroll_topleft) {
            scroll_topleft();
        }
    }
}
