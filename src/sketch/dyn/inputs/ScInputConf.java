package sketch.dyn.inputs;

import java.util.Vector;

import sketch.dyn.ScConstructInfo;
import sketch.dyn.synth.ScStack;
import sketch.util.RichString;

/**
 * Common functions on an array of solving inputs. Most importantly, these can
 * be converted to fixed inputs.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScInputConf implements Cloneable {
    public ScSolvingInputGenerator[] solving_inputs;

    public ScInputConf(ScConstructInfo[] input_info, ScStack al, int log_type) {
        solving_inputs = new ScSolvingInputGenerator[input_info.length];
        for (int a = 0; a < input_info.length; a++) {
            solving_inputs[input_info[a].uid()] =
                    new ScSolvingInputGenerator(al, log_type, input_info[a]);
        }
    }

    @Override
    public String toString() {
        String[] as_str = new String[solving_inputs.length];
        for (int a = 0; a < solving_inputs.length; a++) {
            as_str[a] = solving_inputs[a].toString();
        }
        return String.format("ScInputConfiguration[%s]", (new RichString(", "))
                .join(as_str));
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (ScSolvingInputGenerator input : solving_inputs) {
            result *= 9221;
            result += input.values.hashCode();
        }
        return result;
    }

    /** for clone method only */
    protected ScInputConf() {
    }

    public void add_input(int uid, int v) {
        solving_inputs[uid].addInput(v);
    }

    /** copies synthesized inputs to fixed inputs. result is threadsafe / unique */
    public ScFixedInputGenerator[] fixed_inputs() {
        ScFixedInputGenerator[] result =
                new ScFixedInputGenerator[solving_inputs.length];
        for (int a = 0; a < solving_inputs.length; a++) {
            result[a] = solving_inputs[a].createFixedInput();
        }
        return result;
    }

    public boolean set(int uid, int subuid, int v) {
        return solving_inputs[uid].set(subuid, v);
    }

    public int get(int uid, int subuid) {
        return solving_inputs[uid].values.get(subuid);
    }

    public void reset_index() {
        for (ScSolvingInputGenerator input : solving_inputs) {
            input.reset_index();
        }
    }

    public void reset_accessed(int uid, int subuid) {
        solving_inputs[uid].reset_accessed(subuid);
    }

    @SuppressWarnings("unchecked")
    public void copy_from(ScInputConf oracleInputs) {
        for (int a = 0; a < oracleInputs.solving_inputs.length; a++) {
            solving_inputs[a].next = oracleInputs.solving_inputs[a].next;
            solving_inputs[a].values =
                    (Vector<Integer>) oracleInputs.solving_inputs[a].values
                            .clone();
        }
    }
}
