package sketch.dyn.stats;

import java.util.concurrent.atomic.AtomicLong;

public class ScStatsMT extends ScStats {
    protected AtomicLong nrun = new AtomicLong(0);
    protected AtomicLong ncounterexamples = new AtomicLong(0);

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
}
