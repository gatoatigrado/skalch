package sketch.dyn.synth.stack;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import sketch.dyn.BackendOptions;
import sketch.dyn.constructs.ctrls.ScSynthCtrlConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.stack.prefix.ScDefaultPrefix;
import sketch.dyn.synth.stack.prefix.ScPrefixSearchManager;
import sketch.result.ScSynthesisResults;
import sketch.ui.queues.Queue;

/**
 * cloned Lexin's implementation, then modified for
 * <ul>
 * <li>support for oracle values (!!)</li>
 * <li>backtracking / exploration in multiple regions with multithreading</li>
 * </ul>
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScStackSynthesis extends ScSynthesis<ScLocalStackSynthesis> {
    protected ScSynthCtrlConf ctrls;
    protected ScSolvingInputConf oracleInputs;
    protected Queue queue;

    // variables for ScLocalStackSynthesis
    public ScPrefixSearchManager<ScStack> searchManager;
    protected AtomicBoolean gotFirstRun;
    protected AtomicReference<ScStack> firstSolution;

    public ScStackSynthesis(ScDynamicSketchCall<?>[] sketches, BackendOptions beOpts) {
        super(beOpts);
        // initialize backends
        localSynthesis = new ScLocalStackSynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            localSynthesis[a] = new ScLocalStackSynthesis(sketches[a], this, beOpts, a);
        }
    }

    @Override
    public void synthesizeInner(ScSynthesisResults resultsStore) {
        ScDefaultPrefix prefix = new ScDefaultPrefix();
        ScStack stack = new ScStack(prefix, beOpts.synthOpts.maxStackDepth);
        // shared classes to synchronize / manage search
        searchManager = new ScPrefixSearchManager<ScStack>(stack, prefix);
        gotFirstRun = new AtomicBoolean(false);
        firstSolution = new AtomicReference<ScStack>(null);
        for (ScLocalStackSynthesis localSynth : localSynthesis) {
            resultsStore.addStackSynthesis(localSynth);
            localSynth.run();
        }
    }

    public boolean addSolution(ScStack stack) {
        if (stack.firstRun && gotFirstRun.getAndSet(true)) {
            return false;
        }
        if (firstSolution.get() == null) {
            firstSolution.compareAndSet(null, stack.clone());
        }
        resultsStore.addStackSolution(stack);
        incrementNumSolutions();
        return true;
    }

    @Override
    public Object getSolutionTuple() {
        if (firstSolution.get() == null) {
            return null;
        } else {
            ScStack stack = firstSolution.get();
            stack.resetBeforeRun(); // let the solution just dequeue entries
            return new scala.Tuple2<ScSynthCtrlConf, ScSolvingInputConf>(stack.ctrlConf,
                    stack.oracleConf);
        }
    }
}
