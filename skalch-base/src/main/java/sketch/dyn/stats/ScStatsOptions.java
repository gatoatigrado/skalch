package sketch.dyn.stats;

import sketch.util.cli.CliAnnotatedOptionGroup;
import sketch.util.cli.CliParameter;

public class ScStatsOptions extends CliAnnotatedOptionGroup {
    public ScStatsOptions() {
        super("stat", "statistics options");
    }

    @CliParameter(help = "show entries with zero hits")
    public boolean showZero;
}
