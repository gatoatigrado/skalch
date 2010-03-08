package sketch.dyn.main.angelic;

import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScGeneralDebugEntry;
import sketch.dyn.main.debug.ScLocationDebugEntry;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.queues.QueueIterator;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceLocation;

/**
 * New base Java class for angelic sketches, i.e. sketches that provide test cases and can
 * use the oracle.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScAngelicSketchBase {
    public ScCtrlConf ctrlConf;
    public ScInputConf oracleConf;
    public Vector<ScSourceConstruct> constructSrcInfo = new Vector<ScSourceConstruct>();
    public Vector<Object> sketchQueue;
    public Vector<Object> sketchQueueTrace;
    public QueueIterator queueIterator;

    public boolean debugPrintEnable = false;
    public Vector<ScDebugEntry> debugOut;
    public ScSourceLocation dysketchFcnLocation;
    public int solutionCost = 0;
    public int numAssertsPassed = 0;
    public StackTraceElement debugAssertFailureLocation;
    protected ScSynthesisAssertFailure assertInst__ = new ScSynthesisAssertFailure();
    protected ScDynamicUntilvException untilvInst__ = new ScDynamicUntilvException();

    @Override
    public String toString() {
        return "ScAngelicSketchBase [ctrl_conf=" + ctrlConf + ", oracle_conf=" +
                oracleConf + "]";
    }

    public void synthAssert(boolean truth) {
        if (!truth) {
            if (debugPrintEnable) {
                debugAssertFailureLocation = (new Exception()).getStackTrace()[1];
            }
            throw assertInst__;
        }
        numAssertsPassed += 1;
    }

    public void dynamicUntilvAssert(boolean truth) {
        if (!truth) {
            if (debugPrintEnable) {
                debugAssertFailureLocation = (new Exception()).getStackTrace()[1];
            }
            throw untilvInst__;
        }
    }

    public void enableDebug() {
        debugPrintEnable = true;
        debugAssertFailureLocation = null;
        debugOut = new Vector<ScDebugEntry>();
        sketchQueue = new Vector<Object>();
        sketchQueueTrace = new Vector<Object>();
    }

    public synchronized void skCompilerAssertInternal(Object... arr) {
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure]", "\n", false,
                "FAILED COMPILER ASSERT");
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure]", "\n", false, arr);
        (new Exception()).printStackTrace();
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure] - oracles:", "\n",
                false, oracleConf.toString());
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure] - ctrls:", "\n",
                false, ctrlConf.toString());
        DebugOut.assertFalse("compiler failure");
    }

    public void skCompilerAssert(boolean truth, Object... arr) {
        if (!truth) {
            skCompilerAssertInternal(arr);
        }
    }

    public synchronized void skprint(String... text) {
        DebugOut.print_colored(DebugOut.BASH_GREY, "[program]", " ", false,
                (Object[]) text);
    }

    public void skAddCost(int cost) {
        solutionCost += cost;
    }

    public void skdprintBackend(String text) {
        debugOut.add(new ScGeneralDebugEntry(text));
    }

    public void skqueuePutBackend(int queueNum, Object value) {
        sketchQueue.add(value);
    }

    public void skqueueCheckBackend(int queueNum, Object value, boolean ifDebug) {
        if (queueIterator != null && !queueIterator.checkValue(value)) {
            synthAssert(false);
        }
        if (ifDebug) {
            sketchQueueTrace.add(value);
        }
    }

    public void skdprintLocationBackend(String location) {
        debugOut.add(new ScLocationDebugEntry(location));
    }

    public void addSourceInfo(ScSourceConstruct info) {
        constructSrcInfo.add(info);
    }
}
