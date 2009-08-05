package sketch.dyn.main.debug;

import java.util.Vector;

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;

/**
 * default actions knowing about the Sketch class (versus just the sketch call,
 * which doesn't have a lot of debug info).
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDefaultDebugRun extends ScDebugRun {
    public final ScAngelicSketchBase sketch;

    public ScDefaultDebugRun(
            ScDynamicSketchCall<ScAngelicSketchBase> sketch_call)
    {
        super(sketch_call);
        sketch = sketch_call.get_sketch();
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
        return sketch.debug_out;
    }
}
