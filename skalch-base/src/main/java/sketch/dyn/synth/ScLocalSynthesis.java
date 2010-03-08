package sketch.dyn.synth;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.assertFalse;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.BackendOptions;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.stats.ScStatsMT;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;
import sketch.util.thread.AsyncMTEvent;
import ec.util.MersenneTwisterFast;

public abstract class ScLocalSynthesis implements ScUiQueueable {
    protected ScDynamicSketchCall<?> sketch;
    public AbstractSynthesisThread thread;
    public AsyncMTEvent doneEvents = new AsyncMTEvent();
    // ui
    public final int uid;
    public ConcurrentLinkedQueue<ScUiModifier> uiQueue;
    public boolean animated;
    public final BackendOptions beOpts;

    public ScLocalSynthesis(ScDynamicSketchCall<?> sketch,
            BackendOptions beOpts, int uid)
    {
        this.sketch = sketch;
        this.beOpts = beOpts;
        this.uid = uid;
        animated = beOpts.uiOpts.displayAnimated;
        if (animated && uid > 0) {
            assertFalse("use one thread with the animated option.");
        }
    }

    protected abstract AbstractSynthesisThread createSynthThread();

    public final void run() {
        uiQueue = new ConcurrentLinkedQueue<ScUiModifier>();
        doneEvents.reset();
        if (thread != null && thread.isAlive()) {
            DebugOut.assertFalse("localsynthesis thead alive");
        }
        thread = createSynthThread();
        thread.start();
    }

    public final void threadWait() {
        try {
            thread.join();
            uiQueue = null;
        } catch (InterruptedException e) {
            DebugOut.assertFalse("interrupted waiting for ScLocalSynthesis");
        }
    }

    public final boolean threadAlive() {
        return (thread != null) && (thread.isAlive());
    }

    public final static int NUM_BLIND_FAST = 8192;

    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (uiQueue == null) {
            throw new ScUiQueueableInactive();
        }
        uiQueue.add(m);
    }

    public abstract class AbstractSynthesisThread extends Thread {
        protected MersenneTwisterFast mtLocal;
        protected int nruns = 0, ncounterexamples = 0, nsolutions = 0;
        protected int numCounterexamples = sketch.getNumCounterexamples();

        public AbstractSynthesisThread() {
            if (numCounterexamples <= 0) {
                assertFalse("no counterexamples!");
            }
        }

        protected void updateStats() {
            ScStatsMT.statsSingleton.nruns.add(nruns);
            ScStatsMT.statsSingleton.ncounterexamples.add(ncounterexamples);
            ScStatsMT.statsSingleton.nsolutions.add(nsolutions);
            nruns = 0;
            ncounterexamples = 0;
            nsolutions = 0;
        }

        protected abstract void runInner();

        protected abstract void processUiQueue(ScUiModifier uiModifier);

        @Override
        public final void run() {
            try {
                BackendOptions.backendOpts.set(beOpts);
                mtLocal = mt();
                runInner();
            } catch (ScSynthesisCompleteException e) {
            }
            try {
                while (true) {
                    processUiQueue(uiQueue.remove());
                }
            } catch (NoSuchElementException e) {
                updateStats();
                uiQueue = null;
                doneEvents.set_done();
            }
        }
    }
}
