package skalch;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Add this parameter as a compiler unique identifier. The variable won't appear
 * in the type signature.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface CompilerUid {
}
