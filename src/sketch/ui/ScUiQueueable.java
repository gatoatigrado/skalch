package sketch.ui;

/**
 * A target of a UI Modifier, which will eventually call setInfo on the
 * modifier.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public interface ScUiQueueable {
    public void queueModifier(ScUiModifier m) throws ScUiQueueableInactive;
}
