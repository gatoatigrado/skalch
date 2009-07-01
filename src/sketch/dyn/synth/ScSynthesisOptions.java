package sketch.dyn.synth;

import sketch.util.CliOptGroup;

/**
 * command line options for synthesis.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesisOptions extends CliOptGroup {
    public ScSynthesisOptions() {
        prefixes("sy", "synth", "synthesis");
        add("--num_solutions", -1, "number of solutions to find");
        add("--num_threads", Runtime.getRuntime().availableProcessors(),
                "override number of threads (default # of processors)");
        add("--debug_stop_after", -1, "stop after a number of runs");
        add("--no_clock_rand",
                "don't seed the random number generator with the clock.");
    }
}
