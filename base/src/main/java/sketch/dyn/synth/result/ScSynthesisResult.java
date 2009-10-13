package sketch.dyn.synth.result;

import java.util.LinkedList;

/**
 * This file will be removed soon.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScSynthesisResult {
    public LinkedList<ScSynthesisResult> inferior_results = new LinkedList<ScSynthesisResult>();

    protected ScSynthesisResult fold_left_inner(ScSynthesisResult other) {
        other.addAsInferiorTo(this);
        inferior_results.add(other);
        return this;
    }

    public ScSynthesisResult fold_left(ScSynthesisResult other) {
        if (other == null) {
            return this;
        }
        boolean other_inferior = this.queryArgIsInferior(other);
        if (other_inferior) {
            return this.fold_left_inner(other);
        } else {
            return other.fold_left_inner(this);
        }
    }

    // a result will be inferior if e.g. another thread's result is more
    // important.
    // for example, a thread could observe that the search space is exhausted
    // because another thread grabbed the correct result but didn't set
    // synthesis_complete.
    public abstract boolean inferiorSearchExhausted();

    public abstract boolean inferiorResultFound();

    public void addInferiorSearchExhausted(
            ScSynthesisResultSearchExhausted inferior) {
    }

    public void addInferiorResultFound(ScSynthesisResultFound other) {
    }

    /** return true if $other$ is inferior to $this$ */
    public abstract boolean queryArgIsInferior(ScSynthesisResult other);

    public abstract void addAsInferiorTo(ScSynthesisResult superior);
}
