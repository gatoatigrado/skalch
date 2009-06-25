package sketch.dyn.stats;

public class ScNullStats extends ScStats {
    @Override
    public boolean is_null() {
        return true;
    }

    @Override
    public void run_test(long nruns) {
    }

    @Override
    public void try_counterexample(long ncounterexamples) {
    }

    @Override
    public long num_run_test() {
        throw new java.lang.RuntimeException("trying to query null stats");
    }

    @Override
    public long num_try_counterexample() {
        throw new java.lang.RuntimeException("trying to query null stats");
    }

    @Override
    public void start_synthesis() {
    }

    @Override
    public void stop_synthesis() {
    }

    @Override
    public long get_synthesis_time() {
        throw new java.lang.RuntimeException("trying to query null stats");
    }
}
