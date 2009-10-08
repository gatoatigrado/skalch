package skalch.plugins

object ScalaDebugOut {
    def assert(truth : Boolean, text : => String) : Null = {
        scala.Predef.assert(truth)
        null
    }
    def assertFalse(values : Object*) : Null = {
        println(("" /: values)(_ + "\n" + _))
        scala.Predef.assert(false)
        null
    }
    def not_implemented(values : Object*) : Null = {
        println("[Not implemented] " + ("" /: values)(_ + "\n" + _))
        scala.Predef.assert(false)
        null
    }
}
