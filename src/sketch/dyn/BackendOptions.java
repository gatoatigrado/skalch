package sketch.dyn;

import sketch.util.CliOptGroup;
import sketch.util.CliParser;
import sketch.util.OptionResult;

public class BackendOptions extends CliOptGroup {
    public BackendOptions() {
        prefixes("sk", "sketch");
        add("--num_solutions", 1, "number of solutions to find");
        add("--num_threads", "number of threads (currently unsupported)");
    }

    // java's not quite as concise as Scala
    public static OptionResult create_and_parse(CliParser p) {
        return (new BackendOptions()).parse(p);
    }
}
