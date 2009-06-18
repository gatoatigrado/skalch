package sketch.dyn.stats;

public class ScStatsSeq extends ScStatsMT {
    public long nrun = 0;
    public long ncounterexample = 0;

    @Override
    public void run_test() {
        nrun += 1;
    }

    @Override
    public void try_counterexample() {
        ncounterexample += 1;
    }

    @Override
    public long num_run_test() {
        return nrun;
    }

    @Override
    public long num_try_counterexample() {
        return ncounterexample;
    }
}
