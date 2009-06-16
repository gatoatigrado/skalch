package sketch.dyn.synth.result;

/**
 * This file will be removed soon.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesisResultSearchExhausted extends ScSynthesisResult {

    public boolean inferiorSearchExhausted() {
        return false;
    }

    public boolean inferiorResultFound() {
        return true;
    }

    public boolean queryArgIsInferior(ScSynthesisResult other) {
        return other.inferiorSearchExhausted();
    }

    public void addAsInferiorTo(ScSynthesisResult superior) {
        superior.addInferiorSearchExhausted(this);
    }
}
