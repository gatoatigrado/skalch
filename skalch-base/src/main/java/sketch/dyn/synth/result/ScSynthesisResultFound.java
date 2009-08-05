package sketch.dyn.synth.result;

import sketch.dyn.synth.stack.ScStack;
import sketch.util.DebugOut;

/**
 * This file will be removed soon.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesisResultFound extends ScSynthesisResult {
    public ScStack stack;

    public ScSynthesisResultFound(ScStack stack) {
        this.stack = stack;
    }

    @Override
    public String toString() {
        return "ScSynthesisResultFound[stack=" + stack.toString() + "]";
    }

    public void addInferiorSearchExhausted(
            ScSynthesisResultSearchExhausted inferior) {
        DebugOut.print(this, "superior to", inferior);
    }

    public boolean inferiorResultFound() {
        return false;
    }

    public boolean inferiorSearchExhausted() {
        return false;
    }

    public boolean queryArgIsInferior(ScSynthesisResult other) {
        return other.inferiorResultFound();
    }

    public void addAsInferiorTo(ScSynthesisResult superior) {
        superior.addInferiorResultFound(this);
    }
}
