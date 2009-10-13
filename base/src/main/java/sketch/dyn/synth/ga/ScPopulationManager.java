package sketch.dyn.synth.ga;

import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * mt sync for populations
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScPopulationManager {
    ConcurrentLinkedQueue<ScPopulation> populations =
            new ConcurrentLinkedQueue<ScPopulation>();
}
