package sketch.dyn;

import sketch.dyn.ctrls.ScHoleValue;
import sketch.dyn.inputs.ScInputGenerator;

/**
 * Scala classes inherit this, so the Java code can make nice API calls.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScDynamicSketch {
    public abstract ScConstructInfo[] get_hole_info();

    public abstract ScConstructInfo[] get_input_info();

    public abstract ScConstructInfo[] get_oracle_input_list();

    public abstract void dysketch_main();

    // public abstract void forgiving_assert(boolean v);

    public ScHoleValue[] ctrl_values; // always contain the current valuation
    public ScInputGenerator[] input_backend;
    public ScInputGenerator[] oracle_input_backend;
}
