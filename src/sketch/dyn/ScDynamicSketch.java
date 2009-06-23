package sketch.dyn;

import java.util.Vector;

import sketch.dyn.ctrls.ScCtrlSourceInfo;
import sketch.dyn.ctrls.ScHoleValue;
import sketch.dyn.inputs.ScInputGenerator;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.util.DebugOut;
import sketch.util.RichString;

/**
 * Scala classes inherit this, so the Java code can make nice API calls.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDynamicSketch {
    public ScHoleValue[] ctrl_values; // always contain the current valuation
    public ScInputGenerator[] input_backend;
    public ScInputGenerator[] oracle_input_backend;
    public Vector<ScCtrlSourceInfo> ctrl_src_info =
            new Vector<ScCtrlSourceInfo>();
    public ScSourceLocation dysketch_fcn_location;
    public boolean failed__ = false;
    protected ScSynthesisAssertFailure assert_inst__ =
            new ScSynthesisAssertFailure();

    public abstract ScConstructInfo[] get_hole_info();

    public abstract ScConstructInfo[] get_input_info();

    public abstract ScConstructInfo[] get_oracle_input_list();

    public abstract boolean dysketch_main();

    public abstract ScTestGenerator test_generator();

    public void synthAssertTerminal(boolean truth) {
        if (!truth) {
            throw assert_inst__;
        }
    }

    public void synthAssertForgiving(boolean truth) {
        if (!truth) {
            failed__ = true;
        }
    }

    public String solution_str() {
        Vector<String> result = new Vector<String>();
        result.add("=== generic solution_str() function for sketch ===");
        for (ScHoleValue h : ctrl_values) {
            result.add(h.toString());
        }
        for (ScInputGenerator oracle : oracle_input_backend) {
            result.add(oracle.toString());
        }
        return (new RichString("\n")).join(result.toArray(new String[0]));
    }

    public synchronized void print(String... text) {
        DebugOut.print_colored(DebugOut.BASH_GREY, "[program]", " ", false,
                (Object[]) text);
    }

    public void addHoleSourceInfo(ScCtrlSourceInfo info) {
        ctrl_src_info.add(info);
    }
}
