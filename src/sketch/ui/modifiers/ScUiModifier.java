package sketch.ui.modifiers;

import java.util.concurrent.atomic.AtomicInteger;

import sketch.dyn.ga.ScLocalGaSynthesis;
import sketch.dyn.ga.ScLocalGaSynthesis.SynthesisThread;
import sketch.dyn.stack.ScLocalStackSynthesis;
import sketch.dyn.stack.ScStack;
import sketch.dyn.stats.ScStatsMT;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;
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
public final class ScUiModifier {
    public int timestamp;
    private AtomicInteger enqueue_remaining;
    protected ScUserInterface ui;
    public ScUiModifierInner modifier;

    public ScUiModifier(ScUserInterface ui, ScUiModifierInner modifier) {
        timestamp = ui.nextModifierTimestamp();
        this.ui = ui;
        this.modifier = modifier;
    }

    public final void enqueueTo(ScUiQueueable... targets)
            throws ScUiQueueableInactive
    {
        enqueue_remaining = new AtomicInteger(targets.length + 1);
        for (ScUiQueueable target : targets) {
            target.queueModifier(this);
        }
        setInfoComplete();
    }

    public final void setInfoComplete() {
        if (enqueue_remaining.decrementAndGet() == 0) {
            ui.modifierComplete(this);
        }
    }

    public final void setInfo(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.SynthesisThread synth_thread, ScStack stack)
    {
        modifier.setInfo(local_synth, synth_thread, stack);
        setInfoComplete();
    }

    public final void setInfo(ScStatsMT stats) {
        modifier.setInfo(stats);
        setInfoComplete();
    }

    public void setInfo(ScLocalGaSynthesis gaSynthesis, SynthesisThread thread,
            Object nothing)
    {
        DebugOut.not_implemented("setInfo for ga synth");
    }
}
