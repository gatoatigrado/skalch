package skalch;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Overcome a limitation in Java generics -- pass the class of the resulting
 * type to the function.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface CompilerClassOfResult {
}
