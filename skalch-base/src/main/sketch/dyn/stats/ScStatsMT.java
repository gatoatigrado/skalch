package sketch.dyn.stats;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.BASH_RED;
import static sketch.util.DebugOut.BASH_SALMON;
import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.print_colored;

import java.util.LinkedList;
import java.util.concurrent.atomic.AtomicLong;

import sketch.util.ScRichString;

/**
 * the only stats class. update the num_runs and num_counterexamples at a coarse
 * granularity to avoid mt sync overhead.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatsMT {
    public LinkedList<StatEntry> all_stats = new LinkedList<StatEntry>();
    public StatEntry nruns, ncounterexamples, nsolutions, ga_repeated,
            ga_repeated_recent, ga_nmutate, ga_ncrossover,
            ga_selectind_other_same, ga_selectind_other_optimal,
            ga_selectind_selected_optimal;
    private ArtificalStatEntry ga_total_selectind;
    protected long start_time, end_time;
    public static ScStatsMT stats_singleton;

    public ScStatsMT() {
        if (stats_singleton != null) {
            assertFalse("stats created twice.");
        }
        stats_singleton = this;
        nruns = add_stat(new StatEntry("num_tests", "tests"));
        ncounterexamples = add_stat(new StatEntry("num counterexamples"));
        nsolutions = add_stat(new StatEntry("num solutions"));
        ga_repeated = add_stat(new StatEntry("[ga] num same evaluated"));
        ga_repeated_recent =
                add_stat(new StatEntry("[ga] num same evaluated recent"));
        ga_nmutate = add_stat(new StatEntry("[ga] created from mutate only"));
        ga_ncrossover = add_stat(new StatEntry("[ga] created from crossover"));
        ga_selectind_other_same =
                add_stat(new StatEntry("[ga-tnm] other=same", "tnm_same"));
        ga_selectind_other_optimal =
                add_stat(new StatEntry("[ga-tnm] other optimal",
                        "tnm_other_optimal"));
        ga_selectind_selected_optimal =
                add_stat(new StatEntry("[ga-tnm] selected optimal",
                        "tnm_selected_optimal"));
    }

    private StatEntry add_stat(StatEntry entry) {
        all_stats.add(entry);
        return entry;
    }

    public void start_synthesis() {
        start_time = System.currentTimeMillis();
    }

    private void print_line(String line) {
        print_colored(BASH_SALMON, "[stats]", "", false, line);
    }

    private void print_entry(int align, StatEntry entry) {
        if (beopts().stat_opts.show_zero || entry.value > 0) {
            print_line(entry.formatString(align));
        }
    }

    private void print_entries(StatEntry... entries) {
        int align = 0;
        for (StatEntry entry : entries) {
            print_entry(align, entry);
            align = 30;
        }
    }

    private void print_rate(String indent, StatEntry entry, StatEntry base) {
        if (beopts().stat_opts.show_zero || entry.value > 0) {
            print_line(indent + entry.rate_string(base));
        }
    }

    public void stop_synthesis() {
        end_time = System.currentTimeMillis();
        StatEntry time = new StatEntry("time taken", "sec");
        time.value = (get_synthesis_time()) / 1000.f;
        for (StatEntry entry : all_stats) {
            entry.get_value();
        }
        ga_total_selectind =
                new ArtificalStatEntry(ga_selectind_other_optimal.value
                        + ga_selectind_other_same.value
                        + ga_selectind_selected_optimal.value,
                        "total number of tournament choices", "all selectind");
        print_line("=== statistics ===");
        print_entries(nruns, ncounterexamples, nsolutions, ga_repeated,
                ga_repeated_recent, ga_nmutate, ga_ncrossover);
        if (beopts().stat_opts.show_zero || ga_total_selectind.value > 0) {
            print_entries(ga_total_selectind, ga_selectind_other_same,
                    ga_selectind_other_optimal, ga_selectind_selected_optimal);
        }
        print_entry(0, time);
        print_rate("    ", nruns, time);
        print_rate("", ga_repeated, nruns);
        stat_analysis();
    }

    /**
     * print human-useful information, when stat "events" occur
     */
    private void stat_analysis() {
        rate_warning(ga_repeated.rate(nruns) > 0.5f,
                "GA - many repeat evaluations");
        rate_warning(ga_selectind_other_optimal.rate(ga_total_selectind) < 0.1,
                "GA - pareto optimality rarely replaces individual");
        rate_warning(
                ga_selectind_other_optimal.rate(ga_total_selectind) >= 0.5,
                "GA - pareto optimality replaces individual too often "
                        + "(random number generator error?)");
        rate_warning(ga_selectind_other_same.rate(ga_total_selectind) > 0.1,
                "GA - pareto optimality selects other == selected too often "
                        + "(random number generator error?)");
    }

    private void rate_warning(boolean trigger, String string) {
        if (trigger) {
            print_colored(BASH_RED, "[stat-warning]", "", false, string);
        }
    }

    public long get_synthesis_time() {
        return end_time - start_time;
    }

    public class StatEntry {
        protected AtomicLong ctr = new AtomicLong();
        public Float value;
        public String name;
        public String short_name;

        /**
         * @param short_name
         *            name used when printing rate string
         */
        public StatEntry(String name, String short_name) {
            this.name = name;
            this.short_name = short_name;
        }

        public String formatString(int align) {
            return (new ScRichString(name)).lpad(align) + ": "
                    + String.format("%9.1f", value);
        }

        public float rate(StatEntry base) {
            if (value == 0) {
                return 0;
            }
            return value / base.value;
        }

        public StatEntry(String name) {
            this(name, name);
        }

        public float get_value() {
            value = new Float(ctr.get());
            return value;
        }

        public String rate_string(StatEntry base) {
            return short_name + " / " + base.short_name + ": " + rate(base);
        }

        public void add(long v) {
            ctr.addAndGet(v);
        }
    }

    public class ArtificalStatEntry extends StatEntry {
        public ArtificalStatEntry(float value, String name, String short_name) {
            super(name, short_name);
            this.value = value;
        }

        @Override
        public float get_value() {
            return value;
        }
    }
}
