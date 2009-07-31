package skalch.plugins

import sketch.util.DebugOut

object ScalaDebug {
    def assert(truth : Boolean, text : => String) : Null = {
        if (!truth) {
            DebugOut.assertFalse(text)
        }
        null
    }
    def assertFalse(values : Object*) : Null = {
        DebugOut.assertFalse(values)
        null
    }
    def not_implemented(values : Object*) : Null = {
        DebugOut.not_implemented(values)
        null
    }
}
