package sketch.dyn.ga;

import sketch.util.CliOptGroup;

public class ScGaOptions extends CliOptGroup {
    public ScGaOptions() {
        super("ga", "genetic algorithm options");
        add("--enable", "use the genetic algorithm instead of stack synthesis");
        add("--spine_len", 128, "spine length for phenotype map");
    }
}
