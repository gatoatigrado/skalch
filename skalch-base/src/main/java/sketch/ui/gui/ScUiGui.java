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

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiList;
import sketch.ui.modifiers.ScModifierDispatcher;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceTraceVisitor;
import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;

/**
 * kinda because I can't resist using Eclipse's formatter (it also separates generated
 * code). This file is unfortunately long, but it seems nicer with inner classes.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUiGui extends gui_0_1 {
    private static final long serialVersionUID = 6584583626375432139L;
    public ScUiThread uiThread;
    public ScUiList<ScModifierDispatcher> synthCompletions;
    public int numSynthActive = 0;
    public int contextLen;
    public int contextSplitLen;

    @SuppressWarnings("unchecked")
    /** FIXME - this is incredibly annoying */
    public ScUiGui(ScUiThread uiThread) {
        super();
        this.uiThread = uiThread;
        setupKeys();
        contextLen = uiThread.beOpts.uiOpts.contextLen;
        contextSplitLen = uiThread.beOpts.uiOpts.contextSplitLen;
        // java is very annoying
        int maxListLength = uiThread.beOpts.uiOpts.maxListLength;
        synthCompletions =
                new ScUiList<ScModifierDispatcher>(
                        synthCompletionList,
                        (Class<ScModifierDispatcher[]>) (new ScModifierDispatcher[0]).getClass(),
                        maxListLength);
    }

    @SuppressWarnings("serial")
    public abstract class KeyAction extends AbstractAction {
        public abstract void action();

        public final void add(int keyEvent0, int modifiers, boolean keyRelease) {
            KeyStroke myKeystroke =
                    KeyStroke.getKeyStroke(keyEvent0, modifiers, keyRelease);
            getRootPane().getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(myKeystroke,
                    this);
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
                // set a console UI so the stats are dumped there.
                // ScStatsMT.statsSingleton.ui =
                // new ScDebugConsoleUI(uiThread.beOpts, uiThread.sketchCall);
                stopSolver();
                setVisible(false);
                dispose();
                uiThread.set_stop();
            }
        }).add(KeyEvent.VK_ESCAPE, 0, false);
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                stopSolver();
            }
        }).add(KeyEvent.VK_T, KeyEvent.CTRL_MASK, false);
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                ScModifierDispatcher[] selected = synthCompletions.getSelected();
                if (selected.length >= 1) {
                    synthCompletions.selectNext(selected[0]).dispatch();
                }
            }
        }).add(KeyEvent.VK_J, KeyEvent.CTRL_MASK, true);
        (new KeyAction() {
            private static final long serialVersionUID = -5424144807529052326L;

            @Override
            public void action() {
                ScModifierDispatcher[] selected = synthCompletions.getSelected();
                if (selected.length >= 1) {
                    synthCompletions.selectPrev(selected[0]).dispatch();
                }
            }
        }).add(KeyEvent.VK_K, KeyEvent.CTRL_MASK, true);
    }

    // === ui functions ===
    @Override
    protected void synthCompletionSelectionChanged(ListSelectionEvent evt) {
        ScModifierDispatcher[] selected = synthCompletions.getSelected();
        viewSelectionsButton.setEnabled(selected.length == 1);
        acceptButton.setEnabled(selected.length == 1 && selected[0].isAcceptable());
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
    protected void hyperlinkClicked(String linkurl) {}

    @Override
    protected void stopSolver() {
        uiThread.results.synthesisFinished();
    }

    /** this all happens on the UI thread, but it shouldn't be that slow */
    @SuppressWarnings("unchecked")
    public void fillWithStack(ScModifierDispatcher lastDisplayDispatcher, ScStack stack) {
        // get source
        uiThread.lastDisplayDispatcher = lastDisplayDispatcher;
        stack.initializeFixedForIllustration(uiThread.sketchCall);
        StringBuilder result = getSourceWithSynthesisValues();
        result.append("<p style=\"color: #aaaaaa\">Stack view (in case "
                + "there are bugs above or it's less readable)<br />\n");
        result.append(stack.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");
        sourceCodeEditor.setText(result.toString());
        addDebugInfo(new ScDebugStackRun(
                (ScDynamicSketchCall<ScAngelicSketchBase>) uiThread.sketchCall, stack));
        if (!uiThread.beOpts.uiOpts.noScrollTopleft) {
            scrollTopLeft();
        }
    }

    /**
     * get a string builder with html representing the source and filled in values; most
     * work done by add_source_info()
     */
    protected StringBuilder getSourceWithSynthesisValues() {
        HashMap<String, Vector<ScSourceConstruct>> infoByFilename =
                new HashMap<String, Vector<ScSourceConstruct>>();
        for (ScSourceConstruct holeInfo : uiThread.sourceCodeInfo) {
            String f = holeInfo.entireLocation.filename;
            if (!infoByFilename.containsKey(f)) {
                infoByFilename.put(f, new Vector<ScSourceConstruct>());
            }
            infoByFilename.get(f).add(holeInfo);
        }
        ScSourceCache.singleton().addFilenames(infoByFilename.keySet());
        StringBuilder result = new StringBuilder();
        result.append("<html>\n  <head>\n<style>\nbody {\n"
                + "font-size: 12pt;\n}\n</style>\n  </head>\n  "
                + "<body style=\"margin-top: 0px;\">"
                + "<p style=\"margin-top: 0.1em;\">"
                + "color indicates how often values are changed: red "
                + "is very often, yellow is occasionally, blue is never.</p>");
        for (Entry<String, Vector<ScSourceConstruct>> entry : infoByFilename.entrySet()) {
            result.append("\n<p><pre style=\"font-family: serif;\">");
            addSourceInfo(result, entry.getKey(), entry.getValue());
            result.append("</pre></p><hr />");
        }
        return result;
    }

    /** sub-method for the above (getSourceWithSynthesisValues) */
    protected void addSourceInfo(StringBuilder result, String key,
            Vector<ScSourceConstruct> vector)
    {
        ScSourceConstruct[] holeInfoSorted = vector.toArray(new ScSourceConstruct[0]);
        Arrays.sort(holeInfoSorted);
        ScSourceLocation start =
                holeInfoSorted[0].entireLocation.contextBefore(contextLen);
        ScSourceLocation end =
                holeInfoSorted[holeInfoSorted.length - 1].entireLocation.contextAfter(contextLen);
        ScSourceTraceVisitor v = new ScSourceTraceVisitor();
        // starting context
        result.append(v.visitCode(start));
        // visit constructs and all code in between
        for (int a = 0; a < holeInfoSorted.length; a++) {
            result.append(v.visitHoleInfo(holeInfoSorted[a]));
            ScSourceLocation loc = holeInfoSorted[a].entireLocation;
            if (a + 1 < holeInfoSorted.length) {
                ScSourceLocation nextLoc = holeInfoSorted[a + 1].entireLocation;
                ScSourceLocation between = loc.sourceBetween(nextLoc);
                if (between.numLines() >= contextSplitLen) {
                    // split the context
                    result.append(v.visitCode(loc.contextAfter(contextLen)));
                    result.append("</pre>\n<hr /><pre>");
                    result.append(v.visitCode(nextLoc.contextBefore(contextLen)));
                } else {
                    result.append(v.visitCode(loc.sourceBetween(nextLoc)));
                }
            }
        }
        result.append(v.visitCode(end));
    }

    private void scrollTopLeft() {
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
    private void addDebugInfo(ScDebugRun debugRun) {
        debugRun.run();
        //
        StringBuilder debugText = new StringBuilder();
        debugText.append("<html>\n  <head>\n<style>\n" + "body {\nfont-size: 12pt;\n}\n"
                + "ul {\nmargin-left: 20pt;\n}\n</style>\n  </head>" + "\n  <body>\n<ul>");
        LinkedList<String> htmlContexts = new LinkedList<String>();
        htmlContexts.add("body");
        htmlContexts.add("ul");
        for (ScDebugEntry debugEntry : debugRun.debugOut) {
            debugText.append(debugEntry.htmlString(htmlContexts));
        }
        debugText.append("\n</ul>\n");
        if (debugRun.assertFailed()) {
            StackTraceElement assertInfo = debugRun.assertInfo;
            debugText.append(String.format("<p>failure at %s (line %d)</p>",
                    assertInfo.getMethodName(), assertInfo.getLineNumber()));
        } else {
            debugText.append("<p>dysketch_main returned " +
                    (debugRun.succeeded ? "true" : "false") + "</p>");
        }
        debugText.append("  </body>\n</html>\n");
        debugOutEditor.setText(debugText.toString());
    }

    public void disableStopButton() {
        stopButton.setEnabled(false);
    }

    @Override
    protected void changeDisplayedContext(int nLinesContext) {
        contextLen = nLinesContext;
        contextSplitLen = 3 * nLinesContext;
        if (uiThread.lastDisplayDispatcher != null) {
            uiThread.lastDisplayDispatcher.dispatch();
        }
    }
}
