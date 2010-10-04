package skalch

import skalch.cuda._
import skalch.cuda.annotations._

/**
 * Templates for transforming subtrees. Each template is exported to a GXL file,
 * and then imported when needed. Variables in the class bodies are template parameters.
*/

class RewriteTemplates extends AngelicSketch {
    @SkalchIgnoreClass
    abstract class Template(name : String)

    object IntRangeHole extends Template("int range hole") {
        val from_ : Int = -32
        val to_ : Int = 31

        @generator def fcn = {
            val v = (?? : Int) + from_
            assert (v <= to_)
            v
        }
    }

    /// --------------------------------------------------
    ///  object VLArrayType extends Template("variable length array") {
    ///      class clazz[InnerArrayType] {
    ///          val length : Int = 0;
    ///          val ptr : Array[InnerArrayType] @ scRawArray = null;
    ///      }
    ///  }
    /// -------------------------------------------------- 

}
