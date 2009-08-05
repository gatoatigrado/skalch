package sketch.dyn.synth.ga;

import static ec.util.ThreadLocalMT.mt;
import static sketch.util.DebugOut.print;
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
    @CliParameter(help = "enable GA analysis (adds overhead)")
    public boolean analysis;
    @CliParameter(help = "number of recent solutions in array for analysis")
    public int analysis_recent = 16;
    @CliParameter(help = "spine length for phenotype map")
    public int spine_len = 128;
    @CliParameter(help = "number of individuals per population")
    public int population_sz = 8;
    @CliParameter(help = "number of local populations")
    public int num_populations = 16;
    @CliParameter(help = "probability of using mutation only")
    public ScGaParameter prob_clone_mutate = new ScGaParameter(0f, 0.2f, 1f);
    @CliParameter(help = "probability of tournament selection searching for a better individual.")
    public ScGaParameter prob_reselect = new ScGaParameter(0.1f, 0.5f, 0.9f);

    public final class ScGaParameter implements CliOptionType<ScGaParameter> {
        float min;
        public float value;
        float max;

        public ScGaParameter(float min, float default_value, float max) {
            this.min = min;
            value = default_value;
            this.max = max;
        }

        @Override
        public String toString() {
            return "ScGaParameter[" + value + " \\in [" + min + ", " + max
                    + "] ]";
        }

        @Override
        public ScGaParameter clone() {
            return new ScGaParameter(min, value, max);
        }

        public void perturb() {
            float rand_flt =
                    (float) ((0.1 * mt().nextGaussian()) * (max - min));
            value = Math.max(min, Math.min(max, rand_flt + value));
        }

        public ScGaParameter fromString(String value) {
            float next_value = Float.parseFloat(value);
            if (next_value < min || next_value > max) {
                print("WARNING - parameter", next_value, "out of range", this);
            }
            return new ScGaParameter(min, next_value, max);
        }
    }
}
