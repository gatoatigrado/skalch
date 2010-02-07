package skalch_old.simple;

import java.lang.reflect.InvocationTargetException;
import java.util.Vector;

import org.junit.Assert;

/**
 * run Scala main classes
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScalaMainRunner {
    public static void run(String classname, ArgsOption args) throws Throwable {
        try {
            args = new ArgsOption(args, "--ui-no-gui");
            Class<?> cls = ClassLoader.getSystemClassLoader().loadClass(classname);
            cls.getMethod("main", String[].class).invoke(null, (Object) args.asArray());
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

    public static class ArgsOption {
        Vector<String> args;

        public ArgsOption(Object... args) {
            this.args = new Vector<String>();
            for (Object arg : args) {
                if (arg instanceof ArgsOption) {
                    this.args.addAll(((ArgsOption) arg).args);
                } else {
                    this.args.add(arg.toString());
                }
            }
        }

        public String[] asArray() {
            return args.toArray(new String[0]);
        }
    }

    public static ArgsOption one_soln = new ArgsOption("--sy-num-solutions", "1");
    public static ArgsOption ga = new ArgsOption("--sy-solver", "ga");
    public static ArgsOption ga_one_soln = new ArgsOption(ga, one_soln);
}
