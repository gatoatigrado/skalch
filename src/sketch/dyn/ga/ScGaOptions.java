package sketch.dyn.ga;

import sketch.util.cli.CliOptionGroup;
import sketch.util.cli.CliOptionType;

/**
 * options for the genetic algorithm.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaOptions extends CliOptionGroup {
    public ScGaOptions() {
        super("ga", "genetic algorithm options");
        add("--enable", "use the genetic algorithm instead of stack synthesis");
        add("--spine_len", 128, "spine length for phenotype map");
        add_param("--prob_reslect", 0.1f, 0.5f, 0.9f,
                "probability of tournament selection searching for a better individual.");
    }

    void add_param(String name, float min, float default_value, float max,
            String help)
    {
        add(name, new ScGaParameter(min, default_value, max), help);
    }

    public final class ScGaParameter implements CliOptionType<ScGaParameter> {
        float min;
        float value;
        float max;

        public ScGaParameter(float min, float default_value, float max) {
            this.min = min;
            value = default_value;
            this.max = max;
        }

        @Override
        public ScGaParameter clone() {
            return new ScGaParameter(min, value, max);
        }

        public ScGaParameter fromString(String value) {
            return null;
        }
    }
}
