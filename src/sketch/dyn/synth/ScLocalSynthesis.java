package sketch.dyn.synth;

import static ec.util.ThreadLocalMT.mt;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStats;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.AsyncMTEvent;
import sketch.util.DebugOut;
import ec.util.MersenneTwisterFast;

public abstract class ScLocalSynthesis implements ScUiQueueable {
    protected ScDynamicSketch sketch;
    protected ScFixedInputConf[] counterexamples;
    public AbstractSynthesisThread thread;
    public AsyncMTEvent done_events = new AsyncMTEvent();
    // ui
    public int uid;
    public ConcurrentLinkedQueue<ScUiModifier> ui_queue;

    public ScLocalSynthesis(ScDynamicSketch sketch, int uid) {
        this.sketch = sketch;
        this.uid = uid;
    }

    protected abstract void run_inner();

    public final void run(ScSolvingInputConf[] inputs) {
        counterexamples = ScFixedInputConf.from_inputs(inputs);
        ui_queue = new ConcurrentLinkedQueue<ScUiModifier>();
        done_events.reset();
        if (thread != null && thread.isAlive()) {
            DebugOut.assertFalse("localsynthesis thead alive");
        }
        run_inner();
    }

    public final void thread_wait() {
        try {
            thread.join();
            ui_queue = null;
        } catch (InterruptedException e) {
            DebugOut.assertFalse("interrupted waiting for ScLocalSynthesis");
        }
    }

    public final static int NUM_BLIND_FAST = 8192;

    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (ui_queue == null) {
            throw new ScUiQueueableInactive();
        }
        ui_queue.add(m);
    }

    public abstract class AbstractSynthesisThread extends Thread {
        protected MersenneTwisterFast mt_local;
        protected int nruns = 0, ncounterexamples = 0;

        protected void update_stats() {
            ScStats.stats.run_test(nruns);
            ScStats.stats.try_counterexample(ncounterexamples);
            nruns = 0;
            ncounterexamples = 0;
        }

        protected abstract void run_inner();

        protected abstract void process_ui_queue(ScUiModifier ui_modifier);

        @Override
        public final void run() {
            try {
                mt_local = mt();
                run_inner();
            } catch (ScSynthesisCompleteException e) {
            }
            try {
                while (true) {
                    process_ui_queue(ui_queue.remove());
                }
            } catch (NoSuchElementException e) {
                update_stats();
                ui_queue = null;
                done_events.set_done();
            }
        }
    }
}
