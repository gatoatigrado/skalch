package sketch.dyn.stats;

public class ScNullStats extends ScStats {
    @Override
    public boolean is_null() {
        return true;
    }

    @Override
    public void run_test() {
    }

    @Override
    public void try_counterexample() {
    }

    @Override
    public long num_run_test() {
        throw new java.lang.RuntimeException("trying to query null stats");
    }

    @Override
    public long num_try_counterexample() {
        throw new java.lang.RuntimeException("trying to query null stats");
    }
}
