package sketch.dyn.synth;

import sketch.util.DebugOut;
import sketch.util.cli.CliAnnotatedOptionGroup;
import sketch.util.cli.CliOptionType;
import sketch.util.cli.CliParameter;

/**
 * command line options for synthesis.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScSynthesisOptions extends CliAnnotatedOptionGroup {
    public ScSynthesisOptions() {
        super("sy", "synthesis options");
    }

    @CliParameter(help = "solver to use (ga or stack, ga default)")
    public ScSynthSolver solver = new ScSynthSolver("stack");
    @CliParameter(help = "number of solutions to find")
    public long numSolutions = 1L << 50;
    @CliParameter(help = "override number of threads (default # of processors)")
    public int numThreads = Runtime.getRuntime().availableProcessors();
    @CliParameter(help = "stop after a number of runs")
    public long debugStopAfter = 1L << 50;
    @CliParameter(help = "don't seed the random number generator with the clock.")
    public boolean noClockRand;
    @CliParameter(help = "equivalent to throwing synthAssert() if too many controls"
            + " have been accessed.")
    public int maxStackDepth = 1 << 30L;
    @CliParameter(help = "filename to dump output of queues.")
    public String queueFilename = "";
    @CliParameter(help = "filename to retrieve the queues from the last program refinement.")
    public String queueInFilename = "";
    @CliParameter(help = "filename to dump output of traces.")
    public String traceFilename = "";
    @CliParameter(help = "turn on entanglement analysis.")
    public boolean entanglement = false;

    public final class ScSynthSolver implements CliOptionType<ScSynthSolver> {
        public final String type;
        public final boolean isGa;
        public final boolean isStack;

        public ScSynthSolver(String type) {
            this.type = type;
            isGa = type.toLowerCase().equals("ga");
            isStack = type.toLowerCase().equals("stack");
            DebugOut.assertSlow(((isGa ? 1 : 0) + (isStack ? 1 : 0)) == 1,
                    "only one solver is supported for now, "
                            + "please select ga or stack.");
        }

        @Override
        public String toString() {
            return "ScSynthSolver [type=" + type + "]";
        }

        @Override
        public ScSynthSolver clone() {
            return new ScSynthSolver(type);
        }

        public ScSynthSolver fromString(String value) {
            return new ScSynthSolver(value);
        }
    }
}
