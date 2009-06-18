package sketch.dyn.synth;

/**
 * exception raised when calling through the nice assert functions
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 */
public final class ScSynthesisAssertFailure extends RuntimeException {
    private static final long serialVersionUID = -4184978608394059869L;

    @Override
    public synchronized Throwable fillInStackTrace() {
        return this;
    }
}
