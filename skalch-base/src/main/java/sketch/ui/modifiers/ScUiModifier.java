package sketch.ui.modifiers;

import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.print_mt;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.ui.ScUiQueueable;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.ScUserInterface;

/**
 * An object which will modify the user interface. It will be enqueued to objects
 * supporting ScUiQueueable, which will call setInfo() on it. The targets may be in fast
 * inner loops, so it may take them some time before they can call setInfo. After all info
 * has been set, it will be queued to the user interface, which for GUI's represents a
 * thread. This thread will choose the most recent user interface request of a given event
 * type, and cause its apply method to be called. N.B. - this class is typically
 * overridden as an anonymous inner class that is bound to relevant UI variables.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public final class ScUiModifier {
    public int timestamp;
    private AtomicInteger enqueue_remaining;
    protected ScUserInterface ui;
    public ScUiModifierInner modifier;
    public static AtomicLong queued_modifiers = new AtomicLong(0);

    public ScUiModifier(ScUserInterface ui, ScUiModifierInner modifier) {
        timestamp = ui.nextModifierTimestamp();
        this.ui = ui;
        this.modifier = modifier;
        queued_modifiers.incrementAndGet();
    }

    public final void enqueueTo(ScUiQueueable... targets) throws ScUiQueueableInactive {
        enqueue_remaining = new AtomicInteger(targets.length + 1);
        for (ScUiQueueable target : targets) {
            target.queueModifier(this);
        }
        setInfoComplete();
    }

    public final void setInfoComplete() {
        if (enqueue_remaining.decrementAndGet() == 0) {
            ui.modifierComplete(this);
            long remaining = queued_modifiers.decrementAndGet();
            if (remaining < 0) {
                assertFalse("bad bug - modifiers remaining < 0");
            } else if (remaining > 1000) {
                print_mt("WARNING - a large number of unfinished modifiers exist.");
            }
        }
    }

    public final void setInfo(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.ScStackSynthesisThread synth_thread, ScStack stack)
    {
        modifier.setInfo(local_synth, synth_thread, stack);
        setInfoComplete();
    }

    public final void setInfo(ScStatsMT stats) {
        modifier.setInfo(stats);
        setInfoComplete();
    }
}
