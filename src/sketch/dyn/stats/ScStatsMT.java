package sketch.dyn.stats;

import java.util.concurrent.atomic.AtomicLong;

import sketch.util.DebugOut;

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
        float synth_time = (get_synthesis_time()) / 1000.f;
        float ncouterexamples = num_try_counterexample();
        float ntests = num_run_test();
        float tests_per_sec = ntests / synth_time;
        DebugOut.print_colored(DebugOut.BASH_SALMON, "[stats] ", "\n", false,
                "=== statistics ===", "num tests run: " + ntests,
                "    num counterexamples run: " + ncouterexamples,
                "time taken: " + synth_time, "runs / sec: " + tests_per_sec);
    }

    @Override
    public long get_synthesis_time() {
        return end_time - start_time;
    }
}
