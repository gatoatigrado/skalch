package skalch.plugins

abstract class XmlFormatter {
    private def formatXml(name : String, x : Object, ident: String) : String = {
        val (tagName, attributes, subobjects) = formatObjectInner(x)
        val attstr = ("" /: attributes)((x, y : (String, String)) =>
            x + " %s=\"%s\"".format(y._1, y._2))
        "%s<%s name=\"%s\"%s".format(ident, tagName, name, attstr) +
            (if (subobjects.isEmpty) {
                " />\n"
            } else {
                val subtags = ("" /: subobjects)((x, y : (String, Object)) =>
                    x + formatXml(y._1, y._2, ident + "    "))
                ">\n" + subtags + ident + "</" + tagName + ">\n"
            })
    }

    /** override this: provide a function which will return
        *      the tag name
        *      a list of attributes, as a (name, value) tuple
        *      a list of subobjects, as a (name, object) tuple
        */
    protected def formatObjectInner(x : Object) :
        (String, List[(String, String)], List[(String, Object)])

    def formatXml(x : Object) : String = {
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
        formatXml("root_node", x, "")
    }
}
