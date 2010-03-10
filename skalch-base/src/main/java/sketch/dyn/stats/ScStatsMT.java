package sketch.dyn.stats;

import static sketch.util.DebugOut.assertFalse;

import java.util.LinkedList;

import sketch.dyn.BackendOptions;
import sketch.ui.ScUserInterface;

/**
 * A statistics-gathering class which has a singleton field (stats_singleton)
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScStatsMT {
    public LinkedList<ScStatEntry> allStats = new LinkedList<ScStatEntry>();
    public final ScStatEntry nruns, ncounterexamples, nsolutions, gaRepeated,
            gaRepeatedRecent, gaNMutate, gaNCrossover, gaSelectindOtherSame,
            gaSelectindOtherOptimal, gaSelectindSelectedOptimal;
    private ArtificalStatEntry gaTotalSelectind;
    protected long startTime, endTime;
    public ScUserInterface ui;
    public boolean showZero;

    public static ScStatsMT statsSingleton = new ScStatsMT();

    public static void setUI(ScUserInterface ui) {
        statsSingleton.ui = ui;
    }

    public static void setBackendOptions(BackendOptions beOpts) {
        statsSingleton.showZero = beOpts.statOpts.showZero;
    }

    private ScStatsMT() {
        nruns = addStat(new ScStatEntry("num_tests", "tests"));
        ncounterexamples = addStat(new ScStatEntry("num counterexamples"));
        nsolutions = addStat(new ScStatEntry("num solutions"));
        gaRepeated = addStat(new ScStatEntry("[ga] num same evaluated"));
        gaRepeatedRecent = addStat(new ScStatEntry("[ga] num same evaluated recent"));
        gaNMutate = addStat(new ScStatEntry("[ga] created from mutate only"));
        gaNCrossover = addStat(new ScStatEntry("[ga] created from crossover"));
        gaSelectindOtherSame =
                addStat(new ScStatEntry("[ga-tnm] other=same", "tnm_same"));
        gaSelectindOtherOptimal =
                addStat(new ScStatEntry("[ga-tnm] other optimal", "tnm_other_optimal"));
        gaSelectindSelectedOptimal =
                addStat(new ScStatEntry("[ga-tnm] selected optimal",
                        "tnm_selected_optimal"));
        if (statsSingleton != null) {
            assertFalse("stats created twice.");
        }
        showZero = false;
        ui = null;
    }

    private ScStatEntry addStat(ScStatEntry entry) {
        allStats.add(entry);
        return entry;
    }

    public void startSynthesis() {
        startTime = System.currentTimeMillis();
    }

    public void showStatsWithUi() {
        if (ui == null) {
            return;
        }
        endTime = System.currentTimeMillis();
        ScStatEntry time = new ScStatEntry("time taken", "sec");
        time.value = (getSynthesisTime()) / 1000.f;
        for (ScStatEntry entry : allStats) {
            entry.getValue();
        }
        gaTotalSelectind =
                new ArtificalStatEntry(gaSelectindOtherOptimal.value +
                        gaSelectindOtherSame.value + gaSelectindSelectedOptimal.value,
                        "total number of tournament choices", "all selectind");
        ScStatsModifier modifier = new ScStatsModifier(showZero);
        showStats(time, modifier);
        statAnalysis(modifier);
        ui.setStats(modifier);
    }

    public void showStats(ScStatEntry time, ScStatsModifier modifier) {
        modifier.printEntries(nruns, ncounterexamples, nsolutions, gaRepeated,
                gaRepeatedRecent, gaNMutate, gaNCrossover);
        if (showZero || gaTotalSelectind.value > 0) {
            modifier.printEntries(gaTotalSelectind, gaSelectindOtherSame,
                    gaSelectindOtherOptimal, gaSelectindSelectedOptimal);
        }
        modifier.printEntry(0, time);
        modifier.printRate("    ", nruns, time);
        modifier.printRate("", gaRepeated, nruns);
    }

    /**
     * print human-useful information, when stat "events" occur
     */
    private void statAnalysis(ScStatsModifier modifier) {
        modifier.rateWarnGt(gaRepeated, nruns, 0.5f, "GA - many repeat evaluations");
        modifier.rateWarnLt(gaSelectindOtherOptimal, gaTotalSelectind, 0.1f,
                "GA - pareto optimality rarely replaces individual");
        modifier.rateWarnGt(gaSelectindOtherOptimal, gaTotalSelectind, 0.5f,
                "GA - pareto optimality replaces individual too often "
                        + "(random number generator error?)");
        modifier.rateWarnLt(gaSelectindOtherSame, gaTotalSelectind, 0.1f,
                "GA - pareto optimality selects other == selected too often "
                        + "(random number generator error?)");
    }

    public long getSynthesisTime() {
        return endTime - startTime;
    }
}
