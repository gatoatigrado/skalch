package sketch.dyn;

import java.util.Vector;

import sketch.dyn.ctrls.ScCtrlConf;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.ui.sourcecode.ScSourceLocation;
import sketch.util.DebugOut;

/**
 * Scala classes inherit this, so the Java code can make nice API calls.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDynamicSketch {
    public ScCtrlConf ctrl_conf;
    public ScInputConf input_conf;
    public ScInputConf oracle_conf;
    public Vector<ScSourceConstruct> ctrl_src_info =
            new Vector<ScSourceConstruct>();
    public boolean debug_print_enable = false;
    public Vector<String> debug_out;
    public ScSourceLocation dysketch_fcn_location;
    public StackTraceElement debug_assert_failure_location;
    protected ScSynthesisAssertFailure assert_inst__ =
            new ScSynthesisAssertFailure();
    protected ScDynamicUntilvException untilv_inst__ =
            new ScDynamicUntilvException();

    public abstract ScConstructInfo[] get_hole_info();

    public abstract ScConstructInfo[] get_input_info();

    public abstract ScConstructInfo[] get_oracle_input_list();

    public abstract boolean dysketch_main();

    public abstract ScTestGenerator test_generator();

    public void synthAssertTerminal(boolean truth) {
        if (!truth) {
            if (debug_print_enable) {
                debug_assert_failure_location =
                        (new Exception()).getStackTrace()[1];
            }
            throw assert_inst__;
        }
    }

    public void dynamicUntilvAssert(boolean truth) {
        if (!truth) {
            if (debug_print_enable) {
                debug_assert_failure_location =
                        (new Exception()).getStackTrace()[1];
            }
            throw untilv_inst__;
        }
    }

    public void enable_debug() {
        debug_print_enable = true;
        debug_out = new Vector<String>();
    }

    public synchronized void skprint(String... text) {
        DebugOut.print_colored(DebugOut.BASH_GREY, "[program]", " ", false,
                (Object[]) text);
    }

    public void skdprint_backend(String text) {
        debug_out.add(text);
    }

    public void skdprint_pairs_backend(Object... arr) {
        StringBuilder text = new StringBuilder();
        for (int a = 0; a < arr.length;) {
            String name_str = arr[a].toString();
            text.append(name_str);
            if (a + 1 < arr.length) {
                text.append(": ");
                String value_str = arr[a + 1].toString();
                text.append(value_str);
                if (value_str.length() >= 60) {
                    text.append("\n"); // extra newline for long values
                }
                a += 1;
            }
            a += 1;
            text.append("\n");
        }
        debug_out.add(text.toString());
    }

    public void addHoleSourceInfo(ScSourceConstruct info) {
        DebugOut.print(info);
        ctrl_src_info.add(info);
    }
}
