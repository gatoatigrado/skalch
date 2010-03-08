package sketch.dyn.synth.stack;

import java.util.Vector;

import sketch.dyn.BackendOptions;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScLocalSynthesis;
import sketch.dyn.synth.ScSearchDoneException;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
import sketch.ui.queues.Queue;
import sketch.ui.queues.QueueIterator;
import sketch.util.DebugOut;

/**
 * Container for a synthesis thread. The actual thread is an inner class because
 * the threads will die between synthesis rounds, whereas the sketch object
 * doesn't need to be deleted.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocalStackSynthesis extends ScLocalSynthesis {
    protected ScStackSynthesis ssr;
    public int longestStackSize;
    public ScStack longestStack;
    public Vector<ScStack> randomStacks;
    public Queue queue;

    public ScLocalStackSynthesis(ScDynamicSketchCall<?> sketch,
            ScStackSynthesis ssr, BackendOptions beOpts, int uid) {
        super(sketch, beOpts, uid);
        this.ssr = ssr;
    }

    @Override
    public ScStackSynthesisThread createSynthThread() {
        randomStacks = new Vector<ScStack>();
        return new ScStackSynthesisThread();
    }

    public class ScStackSynthesisThread extends AbstractSynthesisThread {
        ScStack stack;
        int numCounterexamples = 0;
        boolean exhausted = false;
        public float replacementProbability = 1.f;

        /**
         * NOTE - keep this in sync with ScDebugSketchRun
         * 
         * @returns true if exhausted (need to wait)
         */
        protected boolean blindFastRoutine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                boolean forcePop = false;
                trycatch: try {
                    stack.resetBeforeRun();
                    QueueIterator queueIterator = null;
                    if (queue != null) {
                        queueIterator = queue.getIterator();
                    }
                    sketch.initializeBeforeAllTests(stack.ctrlConf,
                            stack.oracleConf, queueIterator);
                    nruns += 1;
                    for (int c = 0; c < sketch.getNumCounterexamples(); c++) {
                        ncounterexamples += 1;
                        if (!sketch.runTest(c)) {
                            break trycatch;
                        }
                    }
                    stack.setCost(sketch.getSolutionCost());
                    if (ssr.addSolution(stack)) {
                        nsolutions += 1;
                    }
                    ssr.waitHandler.throwIfSynthesisComplete();
                } catch (ScSynthesisAssertFailure e) {
                } catch (ScDynamicUntilvException e) {
                    forcePop = true;
                }
                // advance the stack (whether it succeeded or not)
                try {
                    if (longestStackSize < stack.stack.size()) {
                        longestStack = stack.clone();
                        longestStackSize = stack.stack.size();
                    }
                    if (mtLocal.nextFloat() < replacementProbability) {
                        addRandomStack();
                    }
                    stack.next(forcePop);
                } catch (ScSearchDoneException e) {
                    DebugOut.print_mt("exhausted local search");
                    return true;
                }
            }
            return false; // not exhausted
        }

        /** add to the random stacks list and remove half of it */
        private void addRandomStack() {
            randomStacks.add(stack.clone());
            if (randomStacks.size() > ssr.maxNumRandom) {
                int length1 = randomStacks.size() / 2;
                for (int c = 0; c < length1; c++) {
                    randomStacks
                            .remove(mtLocal.nextInt(randomStacks.size()));
                }
                replacementProbability /= 2.f;
            }
        }

        @Override
        public void runInner() {
            longestStackSize = -1;
            longestStack = null;
            stack = ssr.searchManager.cloneDefaultSearch();
            for (long a = 0; !ssr.waitHandler.synthesisComplete.get(); a += NUM_BLIND_FAST) {
                if (a >= ssr.debugStopAfter) {
                    ssr.waitHandler.waitExhausted();
                }
                //
                // NOTE to readers: main call
                exhausted = blindFastRoutine();
                updateStats();
                ssr.waitHandler.throwIfSynthesisComplete();
                if (!uiQueue.isEmpty()) {
                    uiQueue.remove().setInfo(ScLocalStackSynthesis.this, this,
                            stack);
                }
                if (exhausted) {
                    ssr.waitHandler.waitExhausted();
                    ssr.waitHandler.throwIfSynthesisComplete();
                    //
                    DebugOut.not_implemented("get next active stack");
                    stack = ssr.searchManager.getActivePrefix();
                }
            }
        }

        @Override
        public void processUiQueue(ScUiModifier uiModifier) {
            uiModifier.setInfo(ScLocalStackSynthesis.this, this, stack);
        }
    }

    @Override
    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (uiQueue == null) {
            throw new ScUiQueueableInactive();
        }
        uiQueue.add(m);
    }
}
