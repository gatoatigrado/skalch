package sketch.dyn.stats;

import static sketch.util.DebugOut.BASH_SALMON;
import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.print_colored;

import java.util.concurrent.atomic.AtomicLong;

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
    protected AtomicLong nsolutions = new AtomicLong(0);
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

    public void num_solutions(int nsolutions) {
        this.nsolutions.addAndGet(nsolutions);
    }

    public void start_synthesis() {
        start_time = System.currentTimeMillis();
    }

    private void print_line(String line) {
        print_colored(BASH_SALMON, "[stats] ", "", false, line);
    }

    public void stop_synthesis() {
        end_time = System.currentTimeMillis();
        float synth_time = (get_synthesis_time()) / 1000.f;
        float ntests = nrun.get();
        float tests_per_sec = ntests / synth_time;
        print_line("=== statistics ===");
        print_line("num tests run: " + ntests);
        print_line("    num counterexamples run: " + ncounterexamples.get());
        print_line("    num solutions: " + nsolutions.get());
        print_line("time taken: " + synth_time);
        print_line("    runs / sec: " + tests_per_sec);
    }

    public long get_synthesis_time() {
        return end_time - start_time;
    }

    public static class StatEntry {
        public AtomicLong ctr = new AtomicLong();
        public String name;

        public StatEntry(String name) {
            this.name = name;
        }
    }
}
