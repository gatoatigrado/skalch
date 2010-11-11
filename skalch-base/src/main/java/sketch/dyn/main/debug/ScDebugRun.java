package sketch.dyn.main.debug;

import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.queues.QueueIterator;
import sketch.util.DebugOut;

/**
 * Debug run for stack or genetic algorithm synthesis <br />
 * NOTE - keep this in sync with ScLocalStackSynthesis
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScDebugRun {
    protected ScDynamicSketchCall<?> sketchCall;
    public boolean succeeded;
    public StackTraceElement assertInfo;
    public Vector<ScDebugEntry> debugOut;
    public Vector<Object> queue;
    public Vector<Object> queueTrace;

    public ScDebugRun(ScDynamicSketchCall<?> sketch) {
        sketchCall = sketch;
    }

    public abstract void runInit();

    /** feel free to change this method if you need more hooks */
    public final void run() {
        enableDebug();
        runInit();
        sketchCall.initializeBeforeAllTests(getCtrlConf(), getOracleConf(),
                getQueueIterator());
        assertInfo = null;
        succeeded = false;
        trycatch: try {
            for (int a = 0; a < sketchCall.getNumCounterexamples(); a++) {
                if (!sketchCall.runTest(a)) {
                    break trycatch;
                }
            }
            succeeded = true;
        } catch (ScSynthesisAssertFailure e) {
            setAssertInfo(getAssertFailureLocation(), e);
        } catch (ScDynamicUntilvException e) {
            setAssertInfo(getAssertFailureLocation(), e);
        } catch (Exception e) {
            DebugOut.print_exception("should not have any other failures", e);
            DebugOut.assertFalse("exiting");
        }
        debugOut = getDebugOut();
        queue = getQueue();
        queueTrace = getQueueTrace();
    }

    protected final void setAssertInfo(StackTraceElement assertInfo, Exception e) {
        if (assertInfo == null) {
            DebugOut.print("assert info null after failure", e);
        }
        this.assertInfo = assertInfo;
    }

    public boolean assertFailed() {
        return assertInfo != null;
    }

    public void trialInit() {}

    public abstract ScCtrlConf getCtrlConf();

    public abstract ScInputConf getOracleConf();

    protected abstract void enableDebug();

    public abstract StackTraceElement getAssertFailureLocation();

    public abstract Vector<ScDebugEntry> getDebugOut();

    public abstract Vector<Object> getQueue();

    public abstract Vector<Object> getQueueTrace();

    public abstract QueueIterator getQueueIterator();
}
