package sketch.dyn.synth;

import java.util.EmptyStackException;
import java.util.Stack;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScCtrlConf;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.prefix.ScLocalPrefix;
import sketch.dyn.prefix.ScPrefix;
import sketch.dyn.prefix.ScPrefixSearch;
import sketch.util.DebugOut;
import sketch.util.RichString;

/**
 * an instance of a backtracking stack search. each instance essentially
 * searches a "connected" subspace of the input. this implies a new stack search
 * can be created to search "halfway" through the search space, to avoid getting
 * stuck with simple examples like:
 * 
 * <pre>
 * int x = oracle()
 * int y = oracle()
 * assert(x &gt; large_number)
 * </pre>
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStack extends ScPrefixSearch {
    protected ScCtrlConf ctrls;
    protected ScInputConf oracle_inputs;
    protected Stack<ScStackEntry> stack = new Stack<ScStackEntry>();
    protected int added_entries = 0;
    protected boolean first_run = true;
    protected ScConstructInfo[] ctrl_info, oracle_info;

    public final static int SYNTH_HOLE_LOG_TYPE = 3;
    public final static int SYNTH_ORACLE_LOG_TYPE = 6;

    public ScStack(ScConstructInfo[] ctrl_info, ScConstructInfo[] oracle_info,
            ScPrefix default_prefix)
    {
        this.ctrl_info = ctrl_info.clone();
        this.oracle_info = oracle_info.clone();
        ctrls = new ScCtrlConf(ctrl_info, this, SYNTH_HOLE_LOG_TYPE);
        oracle_inputs = new ScInputConf(oracle_info, this,
                SYNTH_ORACLE_LOG_TYPE);
        this.current_prefix = default_prefix;
    }

    @Override
    public String toString() {
        String[] stack_formatted = new String[stack.size()];
        for (int a = 0; a < stack.size(); a++) {
            ScStackEntry ent = stack.get(a);
            if (ent.type == SYNTH_HOLE_LOG_TYPE) {
                stack_formatted[a] = String.format("(hole %d, %d)", ent.uid,
                        ctrls.get(ent.uid));
            } else if (ent.type == SYNTH_ORACLE_LOG_TYPE) {
                stack_formatted[a] = String.format("(oracle %d[%d], %d)",
                        ent.uid, ent.subuid, oracle_inputs.get(ent.uid,
                                ent.subuid));
            }
        }
        return "ScStack[ " + (new RichString(" -> ")).join(stack_formatted)
                + " ]";
    }

    public void set_fixed_for_testing(ScDynamicSketch sketch) {
        sketch.ctrl_values = ctrls.fixed_controls();
        sketch.oracle_input_backend = oracle_inputs.fixed_inputs();
    }

    public void set_for_synthesis(ScDynamicSketch sketch) {
        sketch.ctrl_values = ctrls.ssr_holes;
        sketch.oracle_input_backend = oracle_inputs.solving_inputs;
    }

    protected boolean set_stack_ent(ScStackEntry ent, int v) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrls.set(ent.uid, v);
        } else {
            DebugOut.assert_(ent.type == SYNTH_ORACLE_LOG_TYPE,
                    "uknown stack entry type", ent.type);
            return oracle_inputs.set(ent.uid, ent.subuid, v);
        }
    }

    protected int get_stack_ent(ScStackEntry ent) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrls.get(ent.uid);
        } else {
            DebugOut.assert_(ent.type == SYNTH_ORACLE_LOG_TYPE,
                    "uknown stack entry type", ent.type);
            return oracle_inputs.get(ent.uid, ent.subuid);
        }
    }

    protected void next_inner() {
        if (!current_prefix.get_all_searched()) {
            try {
                ScStackEntry last = stack.peek();
                // get the stack exception first
                int next_value;
                if (current_prefix instanceof ScLocalPrefix) {
                    next_value = get_stack_ent(last) + 1;
                } else {
                    next_value = current_prefix.next_value(this);
                    DebugOut.print_mt("got value", next_value, "from prefix",
                            current_prefix, "uid", last.uid);
                }

                if (!set_stack_ent(last, next_value)) {
                    current_prefix.set_all_searched();
                } else {
                    return;
                }
            } catch (EmptyStackException e) {
                throw new ScSearchDoneException();
            }
        }

        // recurse if this subtree is searched.
        reset_accessed(stack.pop());
        current_prefix = current_prefix.get_parent(this);
        next_inner();
    }

    protected void reset_accessed(ScStackEntry prev) {
        if (prev.type == SYNTH_HOLE_LOG_TYPE) {
            ctrls.reset_accessed(prev.uid);
        } else {
            oracle_inputs.reset_accessed(prev.uid, prev.subuid);
        }
    }

    public void next() {
        // DebugOut.print_mt("   next -", this, first_run, added_entries);
        if (first_run) {
            // now at DefaultPrefix
            first_run = false;
        } else if (added_entries > 0) {
            // need to create a LocalPrefix
            current_prefix = current_prefix.add_entries(added_entries);
        }
        next_inner();

        // reset for next run. oracle_input.reset_index() should happen after
        // next_inner()
        added_entries = 0;
        oracle_inputs.reset_index();
    }

    public void add_entry(int type, int uid, int subuid) {
        stack.add(new ScStackEntry(type, uid, subuid));
        added_entries += 1;
    }

    @SuppressWarnings("unchecked")
    public ScPrefixSearch clone() {
        ScStack result = new ScStack(ctrl_info, oracle_info, current_prefix);
        result.stack = (Stack<ScStackEntry>) this.stack.clone();
        result.first_run = this.first_run;
        DebugOut.assert_(added_entries == 0,
                "please run next() before cloning.");
        return result;
    }
}
