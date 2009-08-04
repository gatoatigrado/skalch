package sketch.dyn.main.old;

import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.main.ScConstructInfo;
import sketch.dyn.main.angelic.ScAngelicSketchBase;

/**
 * Scala classes inherit this, so the Java code can make nice API calls.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScOldDynamicSketch extends ScAngelicSketchBase {
    public ScInputConf input_conf;

    @Override
    public String toString() {
        return "=== ScDynamicSketch ===\n    ctrls: " + ctrl_conf
                + "\n    inputs: " + input_conf + "\n    oracles: "
                + oracle_conf;
    }

    public abstract ScConstructInfo[] get_input_info();

    public abstract boolean dysketch_main();

    public abstract ScTestGenerator test_generator();
}
