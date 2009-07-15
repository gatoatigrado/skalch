package sketch.dyn.ga;

import static ec.util.ThreadLocalMT.mt;
import sketch.util.cli.CliAnnotatedOptionGroup;
import sketch.util.cli.CliOptionType;
import sketch.util.cli.CliParameter;

/**
 * options for the genetic algorithm.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaOptions extends CliAnnotatedOptionGroup {
    public ScGaOptions() {
        super("ga", "genetic algorithm options");
    }

    @CliParameter(help = "use the genetic algorithm instead of stack synthesis")
    public boolean enable;
    @CliParameter(help = "spine length for phenotype map")
    public int spine_len = 128;
    @CliParameter(help = "population size.")
    public int max_population_sz = 8;
    @CliParameter(help = "probability of using mutation only")
    public ScGaParameter prob_clone_mutate = new ScGaParameter(0f, 0.2f, 1f);
    @CliParameter(help = "probability of tournament selection searching for a better individual.")
    public ScGaParameter prob_reselect = new ScGaParameter(0.1f, 0.5f, 0.9f);
    @CliParameter(help = "probability of mutating paramaters that differ between genomes "
            + "instead of using one-point crossover.")
    public ScGaParameter prob_crossover_mutate_different =
            new ScGaParameter(0f, 0.8f, 1f);

    public final class ScGaParameter implements CliOptionType<ScGaParameter> {
        float min;
        public float value;
        float max;

        public float perturb() {
            float rand_flt = (1 - 2 * mt().nextFloat()) * (max - min);
            return Math.max(min, Math.min(max, rand_flt + value));
        }

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
            return new ScGaParameter(min, Float.parseFloat(value), max);
        }
    }
}
