package sketch.util;

import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * A thread with the run() method separated into three sections: init(),
 * run_inner(), and finish(). run_inner() will be called continuously until a
 * stop request is enqueued on the thread.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class InteractiveThread extends Thread {
    public Semaphore stop_event = new Semaphore(0);
    public int wait_time_millis;

    public void init() {
    }

    public abstract void run_inner();

    public void finish() {
    }

    public InteractiveThread(float wait_secs) {
        wait_time_millis = (int) (wait_secs * 1000.f);
    }

    @Override
    public final void run() {
        init();
        try {
            while (!stop_event.tryAcquire(wait_time_millis,
                    TimeUnit.MILLISECONDS))
            {
                run_inner();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        finish();
    }

    public void set_stop() {
        stop_event.release();
    }
}
