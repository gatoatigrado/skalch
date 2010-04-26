package sketch.dyn.synth;

import java.util.concurrent.atomic.AtomicLong;

import sketch.dyn.BackendOptions;
import sketch.result.ScSynthesisResults;
import sketch.ui.queues.Queue;
import sketch.ui.queues.QueueFileInput;
import sketch.util.DebugOut;

/**
 * Base for stack and GA synthesis backends.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScSynthesis<LocalSynthType extends ScLocalSynthesis> {
    // command line options
    public long nsolutionsToFind;
    public long debugStopAfter;
    public int maxNumRandom;
    protected LocalSynthType[] localSynthesis;
    public ScExhaustedWaitHandler waitHandler;
    protected AtomicLong nsolutionsFound;
    public final BackendOptions beOpts;
    protected ScSynthesisResults resultsStore;
    public Queue queue;

    public ScSynthesis(BackendOptions beOpts) {
        nsolutionsFound = new AtomicLong(0);
        this.beOpts = beOpts;
        // command line options
        nsolutionsToFind = beOpts.synthOpts.numSolutions;
        debugStopAfter = beOpts.synthOpts.debugStopAfter;
        maxNumRandom = beOpts.uiOpts.maxRandomStacks;

        if (beOpts.synthOpts.queueInFilename != "") {
            QueueFileInput input = new QueueFileInput(beOpts.synthOpts.queueInFilename);
            queue = input.getQueue();
        }
    }

    public final void synthesize(ScSynthesisResults resultsStore) {
        this.resultsStore = resultsStore;
        waitHandler = new ScExhaustedWaitHandler(localSynthesis.length);
        synthesizeInner(resultsStore);
        for (ScLocalSynthesis localSynth : localSynthesis) {
            localSynth.threadWait();
        }
    }

    protected abstract void synthesizeInner(ScSynthesisResults resultsStore);

    protected final void incrementNumSolutions() {
        nsolutionsFound.incrementAndGet();
        if (nsolutionsFound.longValue() == nsolutionsToFind) {
            DebugOut.print_mt("synthesis complete");
            waitHandler.setSynthesisComplete();
        }
    }

    /** will be unpacked by fluffy scala match statement */
    public abstract Object getSolutionTuple();
}
