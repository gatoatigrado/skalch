package sketch.dyn.main.angelic;

import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.debug.ScDebugEntry;
import sketch.dyn.main.debug.ScGeneralDebugEntry;
import sketch.dyn.main.debug.ScLocationDebugEntry;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceLocation;

/**
 * New base Java class for angelic sketches, i.e. sketches that provide test
 * cases and can use the oracle.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScAngelicSketchBase {
    public ScCtrlConf ctrl_conf;
    public ScInputConf oracle_conf;
    public Vector<ScSourceConstruct> construct_src_info =
            new Vector<ScSourceConstruct>();
    public boolean debug_print_enable = false;
    public Vector<ScDebugEntry> debug_out;
    public ScSourceLocation dysketch_fcn_location;
    public int solution_cost = 0;
    public int num_asserts_passed = 0;
    public StackTraceElement debug_assert_failure_location;
    protected ScSynthesisAssertFailure assert_inst__ =
            new ScSynthesisAssertFailure();
    protected ScDynamicUntilvException untilv_inst__ =
            new ScDynamicUntilvException();

    @Override
    public String toString() {
        return "ScAngelicSketchBase [ctrl_conf=" + ctrl_conf + ", oracle_conf="
                + oracle_conf + "]";
    }

    public void synthAssertTerminal(boolean truth) {
        if (!truth) {
            if (debug_print_enable) {
                debug_assert_failure_location =
                        (new Exception()).getStackTrace()[1];
            }
            throw assert_inst__;
        }
        num_asserts_passed += 1;
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
        debug_assert_failure_location = null;
        debug_out = new Vector<ScDebugEntry>();
    }

    public synchronized void skCompilerAssertInternal(Object... arr) {
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure]", "\n",
                false, "FAILED COMPILER ASSERT");
        DebugOut.print_colored(DebugOut.BASH_RED, "[critical failure]", "\n",
                false, arr);
        (new Exception()).printStackTrace();
        DebugOut.print_colored(DebugOut.BASH_RED,
                "[critical failure] - oracles:", "\n", false, oracle_conf
                        .toString());
        DebugOut.print_colored(DebugOut.BASH_RED,
                "[critical failure] - ctrls:", "\n", false, ctrl_conf
                        .toString());
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
        solution_cost += cost;
    }

    public void skdprint_backend(String text) {
        debug_out.add(new ScGeneralDebugEntry(text));
    }

    public void skdprint_location_backend(String location) {
        debug_out.add(new ScLocationDebugEntry(location));
    }

    /*
     * public void skdprint_pairs_backend(Object... arr) { StringBuilder text =
     * new StringBuilder(); for (int a = 0; a < arr.length;) { String name_str =
     * arr[a].toString(); text.append(name_str); if (a + 1 < arr.length) {
     * text.append(": "); String value_str = arr[a + 1].toString();
     * text.append(value_str); if (value_str.length() >= 60) {
     * text.append("\n"); // extra newline for long values } a += 1; } a += 1;
     * text.append("\n"); } debug_out.add(text.toString()); }
     */
    public void addSourceInfo(ScSourceConstruct info) {
        construct_src_info.add(info);
    }
}
