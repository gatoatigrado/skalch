package skalch

/**
 * Templates for transforming subtrees. Each template is exported to a GXL file,
 * and then imported when needed. Variables in the class bodies are template parameters.
*/

class RewriteTemplates extends AngelicSketch {
    abstract class Template(name : String)

    object IntRangeHole extends Template("int range hole") {
        val lower : Int = -32;
        val upper : Int = 31;

        def fcn = {
            val v = (?? : Int) + lower
            assert (v <= upper)
            v
        }
    }
}
