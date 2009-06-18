package sketch.dyn.stats;

import sketch.dyn.BackendOptions;
import sketch.util.DebugOut;

/**
 * most calls should use the singleton $stats$
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScStats {
    public static ScStats stats;

    /** currently set by ScSynthesis */
    public static void initialize() {
        if (BackendOptions.stat_opts.bool_("enable")) {
            if (BackendOptions.stat_opts.bool_("no_mt_safe")) {
                stats = new ScStatsSeq();
            } else {
                DebugOut.print("using mt stats");
                stats = new ScStatsMT();
            }
        } else {
            stats = new ScNullStats();
        }
    }

    public boolean is_null() {
        return false;
    }

    /** called before a test is run (all counterexamples) */
    public abstract void run_test();

    /** called before a specific counterexample is run */
    public abstract void try_counterexample();

    public static void print_if_enabled() {
        if (stats != null && stats.is_null() == false) {
            float synth_time = ((float) stats.get_synthesis_time()) / 1000.f;
            float ncouterexamples = stats.num_try_counterexample();
            float ntests = stats.num_run_test();
            float tests_per_sec = ntests / synth_time;
            DebugOut
                    .print_colored(DebugOut.BASH_SALMON, "[stats] ", "\n",
                            false, "=== statistics ===", "num tests run: "
                                    + ntests, "    num counterexamples run: "
                                    + ncouterexamples, "time taken: "
                                    + synth_time, "runs / sec: "
                                    + tests_per_sec);
        }
    }

    public abstract void start_synthesis();

    public abstract void stop_synthesis();

    // getters
    public abstract long num_run_test();

    public abstract long num_try_counterexample();

    public abstract long get_synthesis_time();
}
