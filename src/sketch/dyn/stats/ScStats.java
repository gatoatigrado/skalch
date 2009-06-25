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
        if (BackendOptions.stat_opts.bool_("disable")) {
            stats = new ScNullStats();
        } else {
            if (BackendOptions.stat_opts.bool_("no_mt_safe")) {
                stats = new ScStatsSeq();
            } else {
                DebugOut.print("using mt stats");
                stats = new ScStatsMT();
            }
        }
    }

    public boolean is_null() {
        return false;
    }

    /** called before a test is run (all counterexamples) */
    public abstract void run_test(long nruns);

    /** called before a specific counterexample is run */
    public abstract void try_counterexample(long ncounterexamples);

    public abstract void start_synthesis();

    public abstract void stop_synthesis();

    // getters
    public abstract long num_run_test();

    public abstract long num_try_counterexample();

    public abstract long get_synthesis_time();
}
