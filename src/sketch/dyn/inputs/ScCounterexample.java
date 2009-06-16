package sketch.dyn.inputs;

import sketch.dyn.ScDynamicSketch;

/**
 * Wrapper for an array of fixed inputs.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScCounterexample {
    ScFixedInputGenerator[] inputs;

    public ScCounterexample(ScFixedInputGenerator[] inputs) {
        this.inputs = inputs;
    }

    public void set_for_sketch(ScDynamicSketch sketch) {
        for (ScFixedInputGenerator input : inputs) {
            input.reset_index();
        }
        sketch.input_backend = inputs;
    }
}
