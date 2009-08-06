package skalch_old.simple;

import org.junit.Assert;

/**
 * run Scala main classes
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScalaMainRunner {
    public static void run(String classname, String... args) {
        try {
            Class<?> cls =
                    ClassLoader.getSystemClassLoader().loadClass(classname);
            cls.getMethod("main", args.getClass()).invoke(null, (Object) args);
        } catch (Exception e) {
            // e.printStackTrace();
            Assert.fail();
        }
    }
}
