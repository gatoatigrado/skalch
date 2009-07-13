package sketch.dyn.debug;

import java.util.Vector;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.util.DebugOut;

/**
 * Debug run for stack or genetic algorithm synthesis <br />
 * NOTE - keep this in sync with ScLocalStackSynthesis
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDebugRun {
    protected ScDynamicSketch sketch;
    protected ScFixedInputConf[] all_counterexamples;
    public boolean succeeded;
    public StackTraceElement assert_info;
    public Vector<ScDebugEntry> debug_out;

    public ScDebugRun(ScDynamicSketch sketch,
            ScFixedInputConf[] all_counterexamples)
    {
        this.sketch = sketch;
        this.all_counterexamples = all_counterexamples;
    }

    public abstract void run_init();

    /** feel free to change this method if you need more hooks */
    public final void run() {
        sketch.solution_cost = 0;
        sketch.enable_debug();
        assert_info = null;
        succeeded = false;
        trycatch: try {
            for (ScFixedInputConf counterexample : all_counterexamples) {
                counterexample.set_input_for_sketch(sketch);
                if (!sketch.dysketch_main()) {
                    break trycatch;
                }
            }
            succeeded = true;
        } catch (ScSynthesisAssertFailure e) {
            set_assert_info(sketch.debug_assert_failure_location, e);
        } catch (ScDynamicUntilvException e) {
            set_assert_info(sketch.debug_assert_failure_location, e);
        } catch (Exception e) {
            DebugOut.print_exception("should not have any other failures", e);
            DebugOut.assertFalse("exiting");
        }
        debug_out = sketch.debug_out;
        sketch.debug_out = null;
    }

    protected final void set_assert_info(StackTraceElement assert_info,
            Exception e)
    {
        if (assert_info == null) {
            DebugOut.assertFalse("assert info null after failure", e);
        }
        this.assert_info = assert_info;
    }

    public boolean assert_failed() {
        return assert_info != null;
    }

    public void trial_init() {
    }
}
