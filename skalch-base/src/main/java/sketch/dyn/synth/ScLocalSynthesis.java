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
    public AsyncMTEvent done_events = new AsyncMTEvent();
    // ui
    public final int uid;
    public ConcurrentLinkedQueue<ScUiModifier> ui_queue;
    public boolean animated;
    public final BackendOptions be_opts;

    public ScLocalSynthesis(ScDynamicSketchCall<?> sketch,
            BackendOptions be_opts, int uid)
    {
        this.sketch = sketch;
        this.be_opts = be_opts;
        this.uid = uid;
        animated = be_opts.ui_opts.display_animated;
        if (animated && uid > 0) {
            assertFalse("use one thread with the animated option.");
        }
    }

    protected abstract AbstractSynthesisThread create_synth_thread();

    public final void run() {
        ui_queue = new ConcurrentLinkedQueue<ScUiModifier>();
        done_events.reset();
        if (thread != null && thread.isAlive()) {
            DebugOut.assertFalse("localsynthesis thead alive");
        }
        thread = create_synth_thread();
        thread.start();
    }

    public final void thread_wait() {
        try {
            thread.join();
            ui_queue = null;
        } catch (InterruptedException e) {
            DebugOut.assertFalse("interrupted waiting for ScLocalSynthesis");
        }
    }

    public final boolean thread_alive() {
        return (thread != null) && (thread.isAlive());
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
        protected int nruns = 0, ncounterexamples = 0, nsolutions = 0;
        protected int num_counterexamples = sketch.get_num_counterexamples();

        public AbstractSynthesisThread() {
            if (num_counterexamples <= 0) {
                assertFalse("no counterexamples!");
            }
        }

        protected void update_stats() {
            ScStatsMT.stats_singleton.nruns.add(nruns);
            ScStatsMT.stats_singleton.ncounterexamples.add(ncounterexamples);
            ScStatsMT.stats_singleton.nsolutions.add(nsolutions);
            nruns = 0;
            ncounterexamples = 0;
            nsolutions = 0;
        }

        protected abstract void run_inner();

        protected abstract void process_ui_queue(ScUiModifier ui_modifier);

        @Override
        public final void run() {
            try {
                BackendOptions.backend_opts.set(be_opts);
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
