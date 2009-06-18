package sketch.dyn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.LinkedList;

import sketch.dyn.inputs.ScInputConf;
import sketch.util.DebugOut;

/**
 * A method to allow the user to generate test cases to be used as
 * counterexamples during synthesis. It's not very professional atm., and should
 * be revised soon.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScTestGenerator {
    protected LinkedList<ScInputConf> inputs;
    protected ScConstructInfo[] input_info;
    protected ScInputConf current_config;
    protected Method set_method = null;

    // === java-side interfaces ===
    public void init(ScConstructInfo[] input_info) {
        this.input_info = input_info;
        inputs = new LinkedList<ScInputConf>();
        for (Method m : this.getClass().getDeclaredMethods()) {
            boolean all_obj = true;
            for (Class<?> type : m.getParameterTypes()) {
                all_obj &= !type.isPrimitive();
            }
            if (m.getName() == "set") {
                set_method = m;
            }
        }
    }

    public ScInputConf[] get_inputs() {
        return inputs.toArray(new ScInputConf[0]);
    }

    // === scala-side interfaces ===
    public void test_case(Object... args) {
        DebugOut.assert_(set_method != null, "    [test generator] "
                + "please provide a method set([emit_test_case_args])");
        current_config = new ScInputConf(input_info, null, 0);
        try {
            set_method.invoke(this, args);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            DebugOut.assert_(false);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            DebugOut.assert_(false);
        } catch (InvocationTargetException e) {
            e.printStackTrace();
            DebugOut.assert_(false);
        }
        inputs.add(current_config);
        current_config = null;
    }

    public void put_input(ScConstructInfo info, int v) {
        current_config.add_input(info.uid(), v);
    }

    public abstract void tests();
}
