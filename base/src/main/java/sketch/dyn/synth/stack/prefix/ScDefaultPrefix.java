package sketch.dyn.synth.stack.prefix;

/**
 * default (implicit) shared prefix.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScDefaultPrefix extends ScSharedPrefix {
    public ScDefaultPrefix() {
        super(0);
    }

    @Override
    public String toString() {
        return "DefaultPrefix";
    }
}
