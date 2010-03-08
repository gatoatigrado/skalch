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
    protected AtomicInteger nExhausted = new AtomicInteger(0);
    protected Semaphore wait = new Semaphore(0);
    private final int numSynthesis;
    public AtomicBoolean synthesisComplete = new AtomicBoolean(false);

    ScExhaustedWaitHandler(int numSynthesis) {
        this.numSynthesis = numSynthesis;
    }

    public void waitExhausted() {
        if (nExhausted.incrementAndGet() >= numSynthesis) {
            DebugOut.print_mt("all exhausted, exiting");
            setSynthesisComplete();
            nExhausted.addAndGet(-numSynthesis);
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

    public void setSynthesisComplete() {
        synthesisComplete.set(true);
        wait.release(numSynthesis - 1);
    }

    public void throwIfSynthesisComplete() {
        if (synthesisComplete.get()) {
            throw new ScSynthesisCompleteException();
        }
    }
}
