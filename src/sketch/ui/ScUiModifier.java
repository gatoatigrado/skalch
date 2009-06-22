package sketch.ui;

import java.util.concurrent.atomic.AtomicInteger;

import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
import sketch.util.DebugOut;

/**
 * An object which will modify the user interface. It will be enqueued to
 * objects supporting ScUiQueueable, which will call setInfo() on it. The
 * targets may be in fast inner loops, so it may take them some time before they
 * can call setInfo. After all info has been set, it will be queued to the user
 * interface, which for GUI's represents a thread. This thread will choose the
 * most recent user interface request of a given event type, and cause its apply
 * method to be called. N.B. - this class is typically overridden as an
 * anonymous inner class that is bound to relevant UI variables.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScUiModifier {
    protected ScUserInterface ui;
    public int timestamp;
    private AtomicInteger enqueue_remaining;

    public ScUiModifier(ScUserInterface ui) {
        this.ui = ui;
        this.timestamp = ui.nextModifierTimestamp();
    }

    public final void enqueueTo(ScUiQueueable... targets) throws ScUiQueueableInactive {
        this.enqueue_remaining = new AtomicInteger(targets.length);
        for (ScUiQueueable target : targets) {
            target.queueModifier(this);
        }
    }

    public final void setInfoComplete() {
        if (enqueue_remaining.decrementAndGet() == 0) {
            ui.modifierComplete(this);
        }
    }

    public abstract void apply();

    public final void setInfo(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.SynthesisThread synth_thread, ScStack stack)
    {
        setInfoInner(local_synth, synth_thread, stack);
        setInfoComplete();
    }

    public final void setInfo(ScStats stats) {
        setInfoInner(stats);
        setInfoComplete();
    }

    /** make sure to make this method threadsafe! */
    public void setInfoInner(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.SynthesisThread synth_thread, ScStack stack)
    {
        DebugOut.assertFalse("don't append UIModifiers to objects "
                + "they don't have a setInfoInner() for");
    }

    /** make sure to make this method threadsafe! */
    public void setInfoInner(ScStats stats) {
        DebugOut.assertFalse("don't append UIModifiers to objects "
                + "they don't have a setInfoInner() for");
    }
}
