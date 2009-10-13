package sketch.dyn.stats;

import static sketch.util.DebugOut.assertFalse;

import java.util.LinkedList;

import sketch.dyn.BackendOptions;
import sketch.ui.ScUserInterface;

/**
 * A statistics-gathering class which has a singleton field (stats_singleton)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatsMT {
    public LinkedList<ScStatEntry> all_stats = new LinkedList<ScStatEntry>();
    public final ScStatEntry nruns, ncounterexamples, nsolutions, ga_repeated,
            ga_repeated_recent, ga_nmutate, ga_ncrossover,
            ga_selectind_other_same, ga_selectind_other_optimal,
            ga_selectind_selected_optimal;
    private ArtificalStatEntry ga_total_selectind;
    protected long start_time, end_time;
    public static ScStatsMT stats_singleton;
    public ScUserInterface ui;
    public boolean showZero;

    public ScStatsMT(ScUserInterface ui, BackendOptions be_opts) {
        if (ui == null) {
            assertFalse("ScStatsMT - parameter $ui$ null");
        }
        this.ui = ui;
        nruns = add_stat(new ScStatEntry("num_tests", "tests"));
        ncounterexamples = add_stat(new ScStatEntry("num counterexamples"));
        nsolutions = add_stat(new ScStatEntry("num solutions"));
        ga_repeated = add_stat(new ScStatEntry("[ga] num same evaluated"));
        ga_repeated_recent =
                add_stat(new ScStatEntry("[ga] num same evaluated recent"));
        ga_nmutate = add_stat(new ScStatEntry("[ga] created from mutate only"));
        ga_ncrossover =
                add_stat(new ScStatEntry("[ga] created from crossover"));
        ga_selectind_other_same =
                add_stat(new ScStatEntry("[ga-tnm] other=same", "tnm_same"));
        ga_selectind_other_optimal =
                add_stat(new ScStatEntry("[ga-tnm] other optimal",
                        "tnm_other_optimal"));
        ga_selectind_selected_optimal =
                add_stat(new ScStatEntry("[ga-tnm] selected optimal",
                        "tnm_selected_optimal"));
        if (stats_singleton != null) {
            assertFalse("stats created twice.");
        }
        showZero = be_opts.stat_opts.show_zero;
        stats_singleton = this;
    }

    private ScStatEntry add_stat(ScStatEntry entry) {
        all_stats.add(entry);
        return entry;
    }

    public void start_synthesis() {
        start_time = System.currentTimeMillis();
    }

    public void showStatsWithUi() {
        end_time = System.currentTimeMillis();
        ScStatEntry time = new ScStatEntry("time taken", "sec");
        time.value = (get_synthesis_time()) / 1000.f;
        for (ScStatEntry entry : all_stats) {
            entry.get_value();
        }
        ga_total_selectind =
                new ArtificalStatEntry(ga_selectind_other_optimal.value
                        + ga_selectind_other_same.value
                        + ga_selectind_selected_optimal.value,
                        "total number of tournament choices", "all selectind");
        ScStatsModifier modifier = new ScStatsModifier(showZero);
        show_stats(time, modifier);
        stat_analysis(modifier);
        ui.setStats(modifier);
    }

    public void show_stats(ScStatEntry time, ScStatsModifier modifier) {
        modifier.print_entries(nruns, ncounterexamples, nsolutions,
                ga_repeated, ga_repeated_recent, ga_nmutate, ga_ncrossover);
        if (showZero || ga_total_selectind.value > 0) {
            modifier.print_entries(ga_total_selectind, ga_selectind_other_same,
                    ga_selectind_other_optimal, ga_selectind_selected_optimal);
        }
        modifier.print_entry(0, time);
        modifier.print_rate("    ", nruns, time);
        modifier.print_rate("", ga_repeated, nruns);
    }

    /**
     * print human-useful information, when stat "events" occur
     */
    private void stat_analysis(ScStatsModifier modifier) {
        modifier.rate_warn_gt(ga_repeated, nruns, 0.5f,
                "GA - many repeat evaluations");
        modifier.rate_warn_lt(ga_selectind_other_optimal, ga_total_selectind,
                0.1f, "GA - pareto optimality rarely replaces individual");
        modifier.rate_warn_gt(ga_selectind_other_optimal, ga_total_selectind,
                0.5f, "GA - pareto optimality replaces individual too often "
                        + "(random number generator error?)");
        modifier.rate_warn_lt(ga_selectind_other_same, ga_total_selectind,
                0.1f,
                "GA - pareto optimality selects other == selected too often "
                        + "(random number generator error?)");
    }

    public long get_synthesis_time() {
        return end_time - start_time;
    }
}
