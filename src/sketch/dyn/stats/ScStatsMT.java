package sketch.dyn.stats;

import java.util.concurrent.atomic.AtomicLong;

public class ScStatsMT extends ScStats {
    protected AtomicLong nrun = new AtomicLong(0);
    protected AtomicLong ncounterexamples = new AtomicLong(0);
    protected long start_time, end_time;

    @Override
    public void run_test() {
        nrun.incrementAndGet();
    }

    @Override
    public void try_counterexample() {
        ncounterexamples.incrementAndGet();
    }

    @Override
    public long num_run_test() {
        return nrun.get();
    }

    @Override
    public long num_try_counterexample() {
        return ncounterexamples.get();
    }

    @Override
    public void start_synthesis() {
        start_time = System.currentTimeMillis();
    }

    @Override
    public void stop_synthesis() {
        end_time = System.currentTimeMillis();
    }

    @Override
    public long get_synthesis_time() {
        return end_time - start_time;
    }
}
