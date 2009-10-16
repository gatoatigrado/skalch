package skalch.plugins

import sketch.util.DebugOut

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
    def not_implemented(values : Object*) : Null = {
        DebugOut.not_implemented(("" /: values)(_ + "\n" + _))
        null
    }
}
