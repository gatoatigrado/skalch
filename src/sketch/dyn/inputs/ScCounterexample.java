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

    @Override
    public String toString() {
        return "counterexample " + hashCode();
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (ScFixedInputGenerator input : inputs) {
            result *= 666;
            result += input.hashCode();
        }
        return result;
    }

    public void set_for_sketch(ScDynamicSketch sketch) {
        for (ScFixedInputGenerator input : inputs) {
            input.reset_index();
        }
        sketch.input_backend = inputs;
    }

    public static ScCounterexample[] from_inputs(ScInputConf[] inputs) {
        ScCounterexample[] result = new ScCounterexample[inputs.length];
        for (int a = 0; a < inputs.length; a++) {
            result[a] = new ScCounterexample(inputs[a].fixed_inputs());
        }
        return result;
    }
}
