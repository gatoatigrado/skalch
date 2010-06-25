package sketch.compiler.parser.gxlimport;

import sketch.util.cli.CliAnnotatedOptionGroup;
import sketch.util.cli.CliParameter;

/**
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GxlOptions extends CliAnnotatedOptionGroup {
    public GxlOptions() {
        super("gxl", "gxl sketching interface options");
    }

    @CliParameter(help = "Disable defaults (keep tmp, keep asserts)")
    public boolean noDefaults = false;
}
