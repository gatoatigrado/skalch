package sketch.dyn;

import sketch.util.CliOptGroup;
import sketch.util.CliParser;
import sketch.util.OptionResult;

public class BackendOptions extends CliOptGroup {
    public BackendOptions() {
        prefixes("sk", "sketch");
        add("--num_solutions", "number of solutions to print");
        add("--num_threads", "number of threads");
        add("--array_len", "length of array");
    }

    // java's not quite as concise as Scala
    public static OptionResult create_and_parse(CliParser p) {
        return (new BackendOptions()).parse(p);
    }
}
