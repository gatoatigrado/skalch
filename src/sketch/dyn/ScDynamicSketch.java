package sketch.dyn;

import java.util.Vector;

import sketch.dyn.ctrls.ScCtrlConf;
import sketch.dyn.ctrls.ScCtrlSourceInfo;
import sketch.dyn.inputs.ScInputGenerator;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.sourcecode.ScSourceLocation;

/**
 * Scala classes inherit this, so the Java code can make nice API calls.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDynamicSketch {
    public ScCtrlConf ctrl_conf;
    public ScInputGenerator[] input_backend;
    public ScInputGenerator[] oracle_input_backend;
    public Vector<ScCtrlSourceInfo> ctrl_src_info =
            new Vector<ScCtrlSourceInfo>();
    public boolean debug_print_enable = false;
    public Vector<String> debug_out;
    public ScSourceLocation dysketch_fcn_location;
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

    public void enable_debug() {
        debug_print_enable = true;
        debug_out = new Vector<String>();
    }

    public synchronized void print(String... text) {
        // DebugOut.print_colored(DebugOut.BASH_GREY, "[program]", " ", false,
        // (Object[]) text);
    }

    public void addHoleSourceInfo(ScCtrlSourceInfo info) {
        ctrl_src_info.add(info);
    }
}
