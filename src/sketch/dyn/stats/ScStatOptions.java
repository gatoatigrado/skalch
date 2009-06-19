package sketch.dyn.stats;

import sketch.util.CliOptGroup;

/**
 * options for stats, set in BackendOptions.add_opts()
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatOptions extends CliOptGroup {
    public ScStatOptions() {
        prefixes("stat", "statistics");
        add("--enable", "enable statistics");
        add("--profile_enable",
                "enable profiling thread (only supported if <profiler>.set_event()"
                        + " calls are made, which slow execution)");
        add("--no_mt_safe",
                "use sequential (non-threadsafe) stats classes (may be faster)");
    }
}
