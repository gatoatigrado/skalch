package sketch.dyn.main.debug;

import static sketch.util.DebugOut.assertFalse;

import java.util.Vector;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.ui.queues.QueueIterator;

/**
 * default actions knowing about the Sketch class (versus just the sketch call,
 * which doesn't have a lot of debug info).
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDefaultDebugRun extends ScDebugRun {
    public final ScAngelicSketchBase sketch;

    public ScDefaultDebugRun(
            ScDynamicSketchCall<ScAngelicSketchBase> sketchCall) {
        super(sketchCall);
        sketch = sketchCall.getSketch();
    }

    @Override
    public void enableDebug() {
        sketch.enableDebug();
    }

    @Override
    public StackTraceElement getAssertFailureLocation() {
        return sketch.debugAssertFailureLocation;
    }

    @Override
    public Vector<ScDebugEntry> getDebugOut() {
        if (sketch.debugOut == null) {
            assertFalse("ScDefaultDebugRun - debug out null");
        }
        return sketch.debugOut;
    }

    @Override
    public Vector<Object> getQueue() {
        return sketch.sketchQueue;
    }

    @Override
    public Vector<Object> getQueueTrace() {
        return sketch.sketchQueueTrace;
    }

    @Override
    public QueueIterator getQueueIterator() {
        return null;
    }
}
