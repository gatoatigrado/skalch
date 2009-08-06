package skalch_old.simple;

import java.lang.reflect.InvocationTargetException;

import org.junit.Assert;

/**
 * run Scala main classes
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScalaMainRunner {
    public static void run(String classname, String... args) throws Throwable {
        try {
            Class<?> cls =
                    ClassLoader.getSystemClassLoader().loadClass(classname);
            cls.getMethod("main", args.getClass()).invoke(null, (Object) args);
        } catch (ClassNotFoundException e) {
            Assert.fail(e.getMessage());
        } catch (IllegalArgumentException e) {
            Assert.fail(e.getMessage());
        } catch (SecurityException e) {
            Assert.fail(e.getMessage());
        } catch (IllegalAccessException e) {
            Assert.fail(e.getMessage());
        } catch (InvocationTargetException e) {
            throw e.getTargetException();
        } catch (NoSuchMethodException e) {
            Assert.fail(e.getMessage());
        }
    }
}
