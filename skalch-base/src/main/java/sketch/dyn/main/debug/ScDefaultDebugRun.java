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
            ScDynamicSketchCall<ScAngelicSketchBase> sketch_call) {
        super(sketch_call);
        sketch = sketch_call.getSketch();
    }

    @Override
    public void enable_debug() {
        sketch.enable_debug();
    }

    @Override
    public StackTraceElement get_assert_failure_location() {
        return sketch.debug_assert_failure_location;
    }

    @Override
    public Vector<ScDebugEntry> get_debug_out() {
        if (sketch.debug_out == null) {
            assertFalse("ScDefaultDebugRun - debug out null");
        }
        return sketch.debug_out;
    }

    @Override
    public Vector<Object> get_queue() {
        return sketch.sketch_queue;
    }

    @Override
    public Vector<Object> get_queue_trace() {
        return sketch.sketch_queue_trace;
    }

    @Override
    public QueueIterator get_queue_iterator() {
        return null;
    }
}
