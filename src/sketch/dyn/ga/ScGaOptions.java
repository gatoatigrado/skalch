package sketch.dyn.ga;

import sketch.util.cli.CliOptionGroup;

public class ScGaOptions extends CliOptionGroup {
    public ScGaOptions() {
        super("ga", "genetic algorithm options");
        add("--enable", "use the genetic algorithm instead of stack synthesis");
        add("--spine_len", 128, "spine length for phenotype map");
    }
}
