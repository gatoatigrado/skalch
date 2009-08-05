package sketch.dyn.main.debug;

import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
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
    protected ScDynamicSketchCall<?> sketch_call;
    public boolean succeeded;
    public StackTraceElement assert_info;
    public Vector<ScDebugEntry> debug_out;

    public ScDebugRun(ScDynamicSketchCall<?> sketch) {
        sketch_call = sketch;
    }

    public abstract void run_init();

    /** feel free to change this method if you need more hooks */
    public final void run() {
        run_init();
        sketch_call.initialize_before_all_tests(get_ctrl_conf(),
                get_oracle_conf());
        assert_info = null;
        succeeded = false;
        trycatch: try {
            for (int a = 0; a < sketch_call.get_num_counterexamples(); a++) {
                if (!sketch_call.run_test(a)) {
                    break trycatch;
                }
            }
            succeeded = true;
        } catch (ScSynthesisAssertFailure e) {
            set_assert_info(get_assert_failure_location(), e);
        } catch (ScDynamicUntilvException e) {
            set_assert_info(get_assert_failure_location(), e);
        } catch (Exception e) {
            DebugOut.print_exception("should not have any other failures", e);
            DebugOut.assertFalse("exiting");
        }
        debug_out = get_debug_out();
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

    public abstract ScCtrlConf get_ctrl_conf();

    public abstract ScInputConf get_oracle_conf();

    public abstract void enable_debug();

    public abstract StackTraceElement get_assert_failure_location();

    public abstract Vector<ScDebugEntry> get_debug_out();
}
