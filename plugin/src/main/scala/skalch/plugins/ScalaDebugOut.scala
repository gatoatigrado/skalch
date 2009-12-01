package skalch.plugins

import sketch.util.DebugOut
import org.python.util.PythonInterpreter

object ScalaDebugOut {
    def assert(truth : Boolean, text : => String) : Null = {
        if (!truth) {
            DebugOut.assertFalse(text)
        }
        null
    }

    def assertFalse(values : Object*) : Null = {
        DebugOut.assertFalse(("" /: values)(_ + "\n" + _))
        null
    }

    /** usage: debug("myvar" -> obj); */
    def debug(vars : Tuple2[String, Object]) = {
        val python = new PythonInterpreter();
        python.set(vars._1, vars._2);
//         for ((name, obj) <- vars) { python.set(name, obj); }
        python.exec("import pdb; pdb.set_trace();");
    }

    def not_implemented(values : Object*) : Null = {
        DebugOut.not_implemented(("" /: values)(_ + "\n" + _))
        null
    }

    def nonnull[T <: AnyRef](x : T) : T = { assert (!(x eq null), "value null"); x }
}
