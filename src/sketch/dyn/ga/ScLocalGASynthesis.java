package sketch.dyn.ga;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScSynthesisCompleteException;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.AsyncMTEvent;
import sketch.util.DebugOut;

public class ScLocalGASynthesis implements ScUiQueueable {
    protected ScDynamicSketch sketch;
    protected ScGASynthesis gasynth;
    protected int uid;
    protected ScFixedInputConf[] counterexamples;
    public SynthesisThread thread;
    public ConcurrentLinkedQueue<ScUiModifier> ui_queue;
    public AsyncMTEvent done_events = new AsyncMTEvent();

    public ScLocalGASynthesis(ScDynamicSketch sketch, ScGASynthesis gasynth,
            int uid)
    {
        this.sketch = sketch;
        this.gasynth = gasynth;
        this.uid = uid;
    }

    public void run(ScSolvingInputConf[] inputs) {
        counterexamples = ScFixedInputConf.from_inputs(inputs);
        ui_queue = new ConcurrentLinkedQueue<ScUiModifier>();
        done_events.reset();
        if (thread != null && thread.isAlive()) {
            DebugOut.assertFalse("localsynthesis thread alive");
        }
        thread = new SynthesisThread();
        thread.start();
    }

    public final static int NUM_BLIND_FAST = 8192;

    public class SynthesisThread extends Thread {
        protected boolean exhausted;
        protected int nruns = 0, ncounterexamples = 0;

        private boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                DebugOut.not_implemented("blind_fast_routine() for ga");
            }
            return false;
        }

        private void run_inner() {
            for (long a = 0; !gasynth.wait_handler.synthesis_complete.get(); a +=
                    NUM_BLIND_FAST)
            {
                if (gasynth.debug_stop_after != -1
                        && a >= gasynth.debug_stop_after)
                {
                    gasynth.wait_handler.wait_exhausted();
                }
                //
                // NOTE to readers: main call
                exhausted = blind_fast_routine();
                update_stats();
                gasynth.wait_handler.throw_if_synthesis_complete();
                if (!ui_queue.isEmpty()) {
                    ui_queue.remove().setInfo(ScLocalGASynthesis.this, this,
                            null);
                }
                if (exhausted) {
                    gasynth.wait_handler.wait_exhausted();
                    gasynth.wait_handler.throw_if_synthesis_complete();
                }
            }
        }

        @Override
        public void run() {
            try {
                run_inner();
            } catch (ScSynthesisCompleteException e) {
            }
            try {
                while (true) {
                    ui_queue.remove().setInfo(ScLocalGASynthesis.this, this,
                            null);
                }
            } catch (NoSuchElementException e) {
                update_stats();
                ui_queue = null;
                done_events.set_done();
            }
        }

        private void update_stats() {
            ScStats.stats.run_test(nruns);
            ScStats.stats.try_counterexample(ncounterexamples);
            nruns = 0;
            ncounterexamples = 0;
        }
    }

    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive {
        if (ui_queue == null) {
            throw new ScUiQueueableInactive();
        }
        ui_queue.add(m);
    }
}
