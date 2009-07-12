package sketch.dyn.stats;

import static sketch.util.DebugOut.assertFalse;

import java.util.concurrent.atomic.AtomicLong;

import sketch.util.DebugOut;

/**
 * the only stats class. update the num_runs and num_counterexamples at a coarse
 * granularity to avoid mt sync overhead.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatsMT {
    protected AtomicLong nrun = new AtomicLong(0);
    protected AtomicLong ncounterexamples = new AtomicLong(0);
    protected long start_time, end_time;
    public static ScStatsMT stats_singleton;

    public ScStatsMT() {
        if (stats_singleton != null) {
            assertFalse("stats created twice.");
        }
        stats_singleton = this;
    }

    public void run_test(long ntests) {
        nrun.addAndGet(ntests);
    }

    public void try_counterexample(long ncounterexamples_) {
        ncounterexamples.addAndGet(ncounterexamples_);
    }

    public long num_run_test() {
        return nrun.get();
    }

    public long num_try_counterexample() {
        return ncounterexamples.get();
    }

    public void start_synthesis() {
        start_time = System.currentTimeMillis();
    }

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

    public long get_synthesis_time() {
        return end_time - start_time;
    }
}
