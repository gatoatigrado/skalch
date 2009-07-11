package sketch.dyn.stats;

import sketch.util.cli.CliOptionGroup;

/**
 * options for stats, set in BackendOptions.add_opts()
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScStatOptions extends CliOptionGroup {
    public ScStatOptions() {
        super("stat", "statistics options");
        add("--disable", "disable statistics");
        add("--no_mt_safe",
                "use sequential (non-threadsafe) stats classes (may be faster)");
    }
}
