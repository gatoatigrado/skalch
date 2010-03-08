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
    public ScSynthCtrlConf ctrlConf;
    public ScSolvingInputConf oracleConf;
    protected FactoryStack<ScStackEntry> stack;
    protected int addedEntries = 0;
    protected boolean firstRun = true;
    public int solutionCost = -1;
    public final int maxStackDepth;
    public final static int SYNTH_HOLE_LOG_TYPE = 3;
    public final static int SYNTH_ORACLE_LOG_TYPE = 6;

    public ScStack(ScPrefix defaultPrefix, int maxStackDepth) {
        this.maxStackDepth = maxStackDepth;
        stack = new FactoryStack<ScStackEntry>(16, new ScStackEntry.Factory());
        ctrlConf = new ScSynthCtrlConf(this, SYNTH_HOLE_LOG_TYPE);
        oracleConf = new ScSolvingInputConf(this, SYNTH_ORACLE_LOG_TYPE);
        currentPrefix = defaultPrefix;
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (ScStackEntry ent : stack) {
            result *= 77;
            result += ent.hashCode();
        }
        result *= 171;
        result += ctrlConf.hashCode();
        result *= 723;
        result += oracleConf.hashCode();
        return result;
    }

    public String[] getStringArrayRep() {
        String[] rv = new String[stack.size()];
        for (int a = 0; a < stack.size(); a++) {
            ScStackEntry ent = stack.get(a);
            rv[a] =
                    "(" + ent.toString() + ", untilv=" + getUntilv(ent) + ", " +
                            getStackEnt(ent) + ")";
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

    public void initializeFixedForIllustration(ScDynamicSketchCall<?> sketchCall) {
        ctrlConf.generateValueStrings();
        ScFixedInputConf fixedOracles = oracleConf.fixedInputs();
        fixedOracles.generateValueStrings();
        sketchCall.initializeBeforeAllTests(ctrlConf, fixedOracles, null);
    }

    public void resetBeforeRun() {
        addedEntries = 0;
        oracleConf.resetIndex();
    }

    protected boolean setStackEnt(ScStackEntry ent, int v) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrlConf.set(ent.uid, v);
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracleConf.set(ent.uid, ent.subuid, v);
        }
    }

    protected int getStackEnt(ScStackEntry ent) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrlConf.getValue(ent.uid);
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracleConf.get(ent.uid, ent.subuid);
        }
    }

    protected int getUntilv(ScStackEntry ent) {
        if (ent.type == SYNTH_HOLE_LOG_TYPE) {
            return ctrlConf.untilv[ent.uid];
        } else {
            if (ent.type != SYNTH_ORACLE_LOG_TYPE) {
                DebugOut.assertFalse("uknown stack entry type", ent.type);
            }
            return oracleConf.untilv[ent.uid].get(ent.subuid);
        }
    }

    protected void nextInner(boolean forcePop) {
        if (!forcePop && !currentPrefix.getAllSearched()) {
            try {
                ScStackEntry last = stack.peek();
                // get the stack exception first
                int nextValue;
                if (currentPrefix instanceof ScLocalPrefix) {
                    nextValue = getStackEnt(last) + 1;
                } else {
                    nextValue = currentPrefix.nextValue();
                    DebugOut.print_mt("got value", nextValue, "from prefix",
                            currentPrefix, "uid", last.uid);
                }
                if (!setStackEnt(last, nextValue)) {
                    currentPrefix.setAllSearched();
                } else {
                    return;
                }
            } catch (EmptyStackException e) {
                throw new ScSearchDoneException();
            }
        }
        // recurse if this subtree is searched.
        resetAccessed(stack.pop());
        currentPrefix = currentPrefix.getParent(this);
        nextInner(false);
    }

    protected void resetAccessed(ScStackEntry prev) {
        if (prev.type == SYNTH_HOLE_LOG_TYPE) {
            ctrlConf.resetAccessed(prev.uid);
        } else {
            oracleConf.resetAccessed(prev.uid, prev.subuid);
        }
    }

    public void next(boolean forcePop) {
        // DebugOut.print_mt("   next -", this, first_run, added_entries);
        if (firstRun) {
            // now at DefaultPrefix
            firstRun = false;
        } else if (addedEntries > 0) {
            // need to create a LocalPrefix
            currentPrefix = currentPrefix.addEntries(addedEntries);
        }
        nextInner(forcePop);
    }

    public void addEntry(int type, int uid, int subuid) {
        if (stack.size() >= maxStackDepth) {
            throw new ScSynthesisAssertFailure();
        }
        stack.push().set(type, uid, subuid);
        addedEntries += 1;
    }

    @Override
    public ScStack clone() {
        ScStack result = new ScStack(currentPrefix, maxStackDepth);
        result.addedEntries = addedEntries;
        result.ctrlConf.copyValuesFrom(ctrlConf);
        result.oracleConf.copyValuesFrom(oracleConf);
        // ScStackEntry types don't explicitly link to holes or oracles.
        result.stack = stack.clone();
        result.firstRun = firstRun;
        result.solutionCost = solutionCost;
        return result;
    }

    public void setCost(int solutionCost) {
        this.solutionCost = solutionCost;
    }
}
