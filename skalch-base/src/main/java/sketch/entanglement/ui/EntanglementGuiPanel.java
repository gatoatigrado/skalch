package sketch.entanglement.ui;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JEditorPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.text.DefaultCaret;

import kodkod.util.ints.IntSet;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.Event;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.TraceSubset;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.entanglement.sat.SubtraceFilter;
import sketch.entanglement.sat.TraceConverter;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceTraceVisitor;
import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;
import entanglement.MaxSupportFinder;
import entanglement.trace.Traces;

public class EntanglementGuiPanel extends EntanglementGuiPanelBase implements
        ListSelectionListener, ActionListener
{
    private static final long serialVersionUID = 1L;
    private final SATEntanglementAnalysis satEA;
    private final SimpleEntanglementAnalysis ea;
    private final Set<Trace> traces;
    private final List<DynAngel> angelOrder;

    private final List<PartitionPanel> partitionPanels;
    private final Map<Trace, ScStack> traceToStack;
    private ScDynamicSketchCall<?> sketch;
    public Set<ScSourceConstruct> sourceCodeInfo;
    private EntanglementGui gui;
    private EntanglementColoring color;

    public EntanglementGuiPanel(EntanglementGui gui, Map<Trace, ScStack> traceToStack,
            ScDynamicSketchCall<?> sketch, Set<ScSourceConstruct> sourceCodeInfo)
    {
        super();
        this.gui = gui;
        this.traceToStack = traceToStack;
        traces = traceToStack.keySet();
        satEA = new SATEntanglementAnalysis(traces);
        ea = new SimpleEntanglementAnalysis(traces);
        this.sketch = sketch;
        this.sourceCodeInfo = sourceCodeInfo;
        color = new EntanglementColoring(ea, satEA);

        partitionPanels = new ArrayList<PartitionPanel>();
        angelOrder = new ArrayList<DynAngel>();

        if (traceToStack.isEmpty()) {
            return;
        }

        Trace t = traces.iterator().next();
        for (Event e : t.getEvents()) {
            angelOrder.add(e.dynAngel);
        }

        JEditorPane debugEditor = getDebugEditorPane();
        ((DefaultCaret) debugEditor.getCaret()).setUpdatePolicy(DefaultCaret.NEVER_UPDATE);
        JEditorPane programEditor = getProgramEditorPane();
        ((DefaultCaret) programEditor.getCaret()).setUpdatePolicy(DefaultCaret.NEVER_UPDATE);

        JButton refineButton = getRefineAngelButton();
        refineButton.setActionCommand("refineAngel");
        refineButton.addActionListener(this);

        refineButton = getRefineTraceButton();
        refineButton.setActionCommand("refineTrace");
        refineButton.addActionListener(this);

        addSummary();
        addPartitions();
        updateOutput();
    }

    private void addSummary() {
        PartitionSummaryPanel panel =
                new PartitionSummaryPanel(traceToStack, angelOrder, satEA, ea, color);
        JScrollPane entanglementPane = getEntanglementPane();
        entanglementPane.getViewport().add(panel);
    }

    public void addPartitions() {
        JScrollPane partitionPane = getPartitionPane();
        JPanel partitionPanePanel = new JPanel();
        partitionPanePanel.setLayout(new BoxLayout(partitionPanePanel, BoxLayout.Y_AXIS));

        partitionPane.getViewport().add(partitionPanePanel);

        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();
        HashMap<DynAngel, Set<DynAngel>> angelToPartition =
                new HashMap<DynAngel, Set<DynAngel>>();

        for (Set<DynAngel> partition : partitions) {
            for (DynAngel angel : partition) {
                angelToPartition.put(angel, partition);
            }
        }

        for (DynAngel angel : angelOrder) {
            Set<DynAngel> partition = angelToPartition.get(angel);
            if (partitions.contains(partition)) {
                partitions.remove(partition);

                PartitionPanel panel = new PartitionPanel(partition, ea);
                panel.getList().addListSelectionListener(this);

                partitionPanePanel.add(panel);
                partitionPanels.add(panel);
            }

        }

        // partitionPanel.add(checkBox);
    }

    public void valueChanged(ListSelectionEvent arg) {
        if (!arg.getValueIsAdjusting()) {
            updateOutput();
        }
    }

    private void updateOutput() {
        Set<Event> unorderedTrace = new HashSet<Event>();
        boolean clearOutput = false;

        for (PartitionPanel panel : partitionPanels) {
            Trace value = panel.getSelectedValue();
            if (value == null) {
                clearOutput = true;
                break;
            }
            unorderedTrace.addAll(value.getEvents());
        }

        if (clearOutput) {

        } else {
            Trace selectedTrace = null;
            for (Trace trace : traces) {
                if (trace.getEvents().containsAll(unorderedTrace) &&
                        trace.size() == unorderedTrace.size())
                {
                    selectedTrace = trace;
                    break;
                }
            }

            if (selectedTrace != null) {

                ScStack stack = traceToStack.get(selectedTrace);
                stack.setPartitionColor(color.getColorMatrix());
                setProgramOutput(stack);
                ScDebugRun debugRun =
                        new ScDebugStackRun(
                                (ScDynamicSketchCall<ScAngelicSketchBase>) sketch, stack);
                debugRun.run();
                setDebugOutput(debugRun);
            }
        }
    }

    private void setDebugOutput(ScDebugRun debugRun) {
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

        JEditorPane debugEditor = getDebugEditorPane();
        debugEditor.setText(debugText.toString());
    }

    private void setProgramOutput(ScStack stack) {

        stack.initializeFixedForIllustration(sketch);
        StringBuilder result = getSourceWithSynthesisValues();
        result.append("<p style=\"color: #aaaaaa\">Stack view (in case "
                + "there are bugs above or it's less readable)<br />\n");
        result.append(stack.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");

        JEditorPane programEditor = getProgramEditorPane();
        programEditor.setMinimumSize(programEditor.getSize());
        programEditor.setText(result.toString());
    }

    /**
     * get a string builder with html representing the source and filled in values; most
     * work done by add_source_info()
     */
    protected StringBuilder getSourceWithSynthesisValues() {
        HashMap<String, Vector<ScSourceConstruct>> infoByFilename =
                new HashMap<String, Vector<ScSourceConstruct>>();
        for (ScSourceConstruct holeInfo : sourceCodeInfo) {
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
        int contextLen = 3;
        int contextSplitLen = 5;

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

    public void actionPerformed(ActionEvent event) {
        if ("refineAngel".equals(event.getActionCommand())) {
            final EntanglementGuiPanel parent = this;

            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    DisentanglementGui gui =
                            new DisentanglementGui(parent, false,
                                    new ArrayList<DynAngel>(angelOrder));
                    gui.setVisible(true);
                }
            });
        } else if ("refineTrace".equals(event.getActionCommand())) {
            Set<Trace> filteredTraces = new HashSet<Trace>();
            for (Trace trace : traces) {
                boolean isBad = false;
                for (PartitionPanel panel : partitionPanels) {
                    Set<Trace> badTraces = panel.getBadTraces();
                    if (badTraces.contains(trace.getSubTrace(panel.getAngelPartition())))
                    {
                        isBad = true;
                        break;
                    }
                }
                if (!isBad) {
                    filteredTraces.add(trace);
                }
            }
        }
    }

    public void partitionTraces(Set<Set<DynAngel>> subpartitioning) {

        TraceConverter converter = satEA.getTraceConverter();
        List<IntSet> oldSatPartitions = satEA.getEntangledIntSets();
        Set<Set<DynAngel>> oldPartitions =
                converter.getDynAngelPartitions(oldSatPartitions);

        Set<Set<DynAngel>> partitioning = new HashSet<Set<DynAngel>>();

        for (Set<DynAngel> partition : oldPartitions) {
            HashSet<DynAngel> partitionClone = new HashSet<DynAngel>(partition);
            for (Set<DynAngel> subpartition : subpartitioning) {
                HashSet<DynAngel> projection = new HashSet<DynAngel>();
                for (DynAngel angel : subpartition) {
                    if (partition.contains(angel)) {
                        projection.add(angel);
                    }
                }
                if (!projection.isEmpty()) {
                    partitioning.add(projection);
                    partitionClone.removeAll(projection);
                }
            }
            if (!partitionClone.isEmpty()) {
                partitioning.add(partitionClone);
            }
        }

        List<IntSet> newPartitions = converter.getIntSetPartitions(partitioning);

        // EntanglementSummaryGui newGui =
        // new EntanglementSummaryGui(sketch, sourceCodeInfo);

        final Set<Trace> goodTraces = new HashSet<Trace>();
        final Set<Trace> badTraces = new HashSet<Trace>();

        for (PartitionPanel panel : partitionPanels) {
            goodTraces.addAll(panel.getGoodTraces());
            badTraces.addAll(panel.getBadTraces());
        }

        Traces satTraces = converter.getTraces();

        boolean updateTraces = !badTraces.isEmpty();

        if (updateTraces) {
            satTraces =
                    satTraces.restrict(new SubtraceFilter(badTraces, converter, true));
        }

        List<TraceSubset> subsets = new ArrayList<TraceSubset>();
        int i = 0;

        for (Iterator<Traces> supports =
                MaxSupportFinder.findMaximalSupports(satTraces, oldSatPartitions,
                        newPartitions); supports.hasNext();)
        {
            Traces support = supports.next();
            List<Trace> subsetTraces = converter.convert(support);
            if (goodSubset(subsetTraces, goodTraces)) {
                subsets.add(new TraceSubset(subsetTraces, "" + i, null));
            }
            i++;
        }

        // newGui.pack();
        // newGui.setVisible(true);

        EntanglementGui gui =
                new EntanglementGui(subsets, traceToStack, sketch, sourceCodeInfo);
        gui.setVisible(true);
    }

    private boolean goodSubset(List<Trace> subsetTraces, Set<Trace> goodSubtraces) {
        for (Trace goodTrace : goodSubtraces) {
            Set<DynAngel> angels = goodTrace.getAngels();
            for (Trace trace : subsetTraces) {
                if (trace.getSubTrace(angels).equals(goodTrace)) {
                    continue;
                } else {
                    return false;
                }
            }
        }

        return true;
    }

    public Frame getFrame() {
        return gui;
    }
}
