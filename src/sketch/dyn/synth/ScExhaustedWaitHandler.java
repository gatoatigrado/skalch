package sketch.dyn.synth;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import sketch.util.DebugOut;

/**
 * class that will make current thread sleep when calling wait_exhausted() and
 * there are other active threads. also determines when synthesis is complete.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScExhaustedWaitHandler {
    private final int num_synthesis;

    ScExhaustedWaitHandler(int num_synthesis) {
        this.num_synthesis = num_synthesis;
    }

    protected AtomicInteger n_exhausted = new AtomicInteger(0);
    protected Semaphore wait = new Semaphore(0);
    public AtomicBoolean synthesis_complete = new AtomicBoolean(false);

    public void wait_exhausted() {
        if (n_exhausted.incrementAndGet() >= num_synthesis) {
            DebugOut.print_mt("all exhausted, exiting");
            set_synthesis_complete();
            n_exhausted.addAndGet(-num_synthesis);
            return;
        }
        DebugOut.print_mt("exhausted handler waiting");
        try {
            wait.acquire();
        } catch (InterruptedException e) {
            e.printStackTrace();
            DebugOut.assertFalse("don't interrupt threads.");
        }
        DebugOut.print_mt("done waiting");
    }

    public void set_synthesis_complete() {
        synthesis_complete.set(true);
        wait.release(num_synthesis - 1);
    }

    public void throw_if_synthesis_complete() {
        if (synthesis_complete.get()) {
            throw new ScSynthesisCompleteException();
        }
    }
}
