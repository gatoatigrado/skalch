package sketch.entanglement.ui.program;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import java.util.Map.Entry;

import javax.swing.JEditorPane;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScDebugRun;
import sketch.dyn.main.debug.ScDebugStackRun;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.Trace;
import sketch.entanglement.ui.EntanglementColoring;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceTraceVisitor;
import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;

public class SkalchDisplay implements ProgramDisplay {
    private Map<Trace, ScStack> traceToStack;
    private ScDynamicSketchCall<?> sketch;
    private Set<ScSourceConstruct> sourceCodeInfo;

    public SkalchDisplay(Map<Trace, ScStack> traceToStack, ScDynamicSketchCall<?> sketch,
            Set<ScSourceConstruct> sourceCodeInfo)
    {
        this.traceToStack = traceToStack;
        this.sketch = sketch;
        this.sourceCodeInfo = sourceCodeInfo;
    }

    public ProgramOutput getProgramOutput(Trace selectedTrace, EntanglementColoring color)
    {
        return new ProgramOutput(getProgramText(selectedTrace, color),
                getDebugOutput(selectedTrace, color));
    }

    public String getDebugOutput(Trace selectedTrace, EntanglementColoring color) {
        ScStack stack = traceToStack.get(selectedTrace);
        stack.setPartitionColor(color.getColorMatrix());
        stack.initializeFixedForIllustration(sketch);

        ScDebugRun debugRun =
                new ScDebugStackRun((ScDynamicSketchCall<ScAngelicSketchBase>) sketch,
                        stack);
        debugRun.run();
        return getDebugText(debugRun);
    }

    private String getDebugText(ScDebugRun debugRun) {
        StringBuilder debugText = new StringBuilder();
        debugText.append("<html>\n  <head>\n<style>\n" + "body {\nfont-size: 12pt;\n}\n"
                + "ul {\nmargin-left: 20pt;\n}\n</style>\n  </head>" + "\n  <body>\n<ul>");

        boolean newLine = true;
        for (ScDebugEntry debugEntry : debugRun.debugOut) {
            if (newLine) {
                debugText.append("<li>");
                newLine = false;
            }
            debugText.append(debugEntry.htmlString());
            if (debugEntry.hasEndline()) {
                debugText.append("</li>\n");
                newLine = true;
            }
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
        return debugText.toString();
    }

    public String getProgramText(Trace selectedTrace, EntanglementColoring color) {
        ScStack stack = traceToStack.get(selectedTrace);
        stack.setPartitionColor(color.getColorMatrix());
        stack.initializeFixedForIllustration(sketch);

        StringBuilder result = getSourceWithSynthesisValues();
        result.append("<p style=\"color: #aaaaaa\">Stack view (in case "
                + "there are bugs above or it's less readable)<br />\n");
        result.append(stack.htmlDebugString());
        result.append("\n</p>\n</body>\n</html>");

        return result.toString();
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
}
