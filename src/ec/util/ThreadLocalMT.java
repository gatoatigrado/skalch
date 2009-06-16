package ec.util;

/**
 * ThreadLocal<MersenneTwisterFast> wrapper with initialValue() function
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ThreadLocalMT extends ThreadLocal<MersenneTwisterFast> {
    public int salt = 0;

    public ThreadLocalMT() {
    }

    public ThreadLocalMT(int salt) {
        this.salt = salt;
    }

    @Override
    protected MersenneTwisterFast initialValue() {
        return new MersenneTwisterFast(Thread.currentThread().getId() + salt);
    }
}
