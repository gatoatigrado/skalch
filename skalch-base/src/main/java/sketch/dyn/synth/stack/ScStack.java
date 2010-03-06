package sketch.dyn.synth.stack;

import java.util.EmptyStackException;

import sketch.dyn.constructs.ctrls.ScSynthCtrlConf;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSearchDoneException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.dyn.synth.stack.prefix.ScLocalPrefix;
import sketch.dyn.synth.stack.prefix.ScPrefix;
import sketch.dyn.synth.stack.prefix.ScPrefixSearch;
import sketch.util.DebugOut;
import sketch.util.datastructures.FactoryStack;
import sketch.util.wrapper.ScRichString;

/**
 * an instance of a backtracking stack search. each instance essentially searches a
 * "connected" subspace of the input. this implies a new stack search can be created to
 * search "halfway" through the search space, to avoid getting stuck with simple examples
 * like:
 * 
 * <pre>
 * int x = oracle()
 * int y = oracle()
 * assert(x &gt; large_number)
 * </pre>
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScStack extends ScPrefixSearch {
    public ScSynthCtrlConf ctrl_conf;
    public ScSolvingInputConf oracle_conf;
    protected FactoryStack<ScStackEntry> stack;
    protected int added_entries = 0;
    protected boolean first_run = true;
    public int solution_cost = -1;
    public final int max_stack_depth;
    public final static int SYNTH_HOLE_LOG_TYPE = 3;
    public final static int SYNTH_ORACLE_LOG_TYPE = 6;

    public ScStack(ScPrefix default_prefix, int max_stack_depth) {
        this.max_stack_depth = max_stack_depth;
        stack = new FactoryStack<ScStackEntry>(16, new ScStackEntry.Factory());
        ctrl_conf = new ScSynthCtrlConf(this, SYNTH_HOLE_LOG_TYPE);
        oracle_conf = new ScSolvingInputConf(this, SYNTH_ORACLE_LOG_TYPE);
        current_prefix = default_prefix;
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (ScStackEntry ent : stack) {
            result *= 77;
            result += ent.hashCode();
        }
        result *= 171;
        result += ctrl_conf.hashCode();
        result *= 723;
        result += oracle_conf.hashCode();
        return result;
    }

    public String[] getStringArrayRep() {
        String[] rv = new String[stack.size()];
        for (int a = 0; a < stack.size(); a++) {
            ScStackEntry ent = stack.get(a);
            rv[a] =
                    "(" + ent.toString() + ", untilv=" + get_untilv(ent) + ", " +
                            get_stack_ent(ent) + ")";
        }
        return rv;
    }

    @Override
    public String toString() {
        return "ScStack[ " + (new ScRichString(" -> ")).join(getStringArrayRep()) + " ]";
    }

    public String htmlDebugString() {
        ScRichString sep = new ScRichString(" -> <br />&nbsp;&nbsp;&nbsp;&nbsp;");
        return "ScStack[ " + sep.join(getStringArrayRep()) + " ]";
    }

    public void initialize_fixed_for_illustration(ScDynamicSketchCall<?> sketch_call) {
        ctrl_conf.generate_value_strings();
        ScFixedInputConf fixed_oracles = oracle_conf.fixed_inputs();
        fixed_oracles.generate_value_strings();
        sketch_call.initialize_before_all_tests(ctrl_conf, fixed_oracles, null);
    }

    public void reset_before_run() {
        added_entries = 0;
        oracle_conf.reset_index();
    }

    protected boolean set_stack_ent(ScStackEntry ent, int v) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrl_conf.set(ent.uid, v);
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracle_conf.set(ent.uid, ent.subuid, v);
        }
    }

    protected int get_stack_ent(ScStackEntry ent) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrl_conf.getValue(ent.uid);
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracle_conf.get(ent.uid, ent.subuid);
        }
    }

    protected int get_untilv(ScStackEntry ent) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrl_conf.untilv[ent.uid];
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracle_conf.untilv[ent.uid].get(ent.subuid);
        }
    }

    protected void next_inner(boolean force_pop) {
        if (!force_pop && !current_prefix.get_all_searched()) {
            try {
                ScStackEntry last = stack.peek();
                // get the stack exception first
                int next_value;
                if (current_prefix instanceof ScLocalPrefix) {
                    next_value = get_stack_ent(last) + 1;
                } else {
                    next_value = current_prefix.next_value();
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
        next_inner(false);
    }

    protected void reset_accessed(ScStackEntry prev) {
        if (prev.type == SYNTH_HOLE_LOG_TYPE) {
            ctrl_conf.reset_accessed(prev.uid);
        } else {
            oracle_conf.reset_accessed(prev.uid, prev.subuid);
        }
    }

    public void next(boolean force_pop) {
        // DebugOut.print_mt("   next -", this, first_run, added_entries);
        if (first_run) {
            // now at DefaultPrefix
            first_run = false;
        } else if (added_entries > 0) {
            // need to create a LocalPrefix
            current_prefix = current_prefix.add_entries(added_entries);
        }
        next_inner(force_pop);
    }

    public void add_entry(int type, int uid, int subuid) {
        if (stack.size() >= max_stack_depth) {
            throw new ScSynthesisAssertFailure();
        }
        stack.push().set(type, uid, subuid);
        added_entries += 1;
    }

    @Override
    public ScStack clone() {
        ScStack result = new ScStack(current_prefix, max_stack_depth);
        result.added_entries = added_entries;
        result.ctrl_conf.copy_values_from(ctrl_conf);
        result.oracle_conf.copy_values_from(oracle_conf);
        // ScStackEntry types don't explicitly link to holes or oracles.
        result.stack = stack.clone();
        result.first_run = first_run;
        result.solution_cost = solution_cost;
        return result;
    }

    public void setCost(int solution_cost) {
        this.solution_cost = solution_cost;
    }
}
