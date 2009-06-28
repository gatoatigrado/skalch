package sketch.util;

import nu.xom.Element;
import nu.xom.Nodes;

/**
 * wrap the nu.xom class with some less verbose functions.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class XmlEltWrapper extends Element {
    public XmlEltWrapper(Element element) {
        super(element);
    }

    public XmlEltWrapper(String element) {
        super(element);
    }

    public XmlEltWrapper(String s1, String s2) {
        super(s1, s2);
    }

    public XmlEltWrapper XpathElt(String querystr) {
        Nodes nodes = query(querystr);
        if (nodes.size() != 1) {
            DebugOut
                    .assertFalse("multiple nodes returned from query", querystr);
        } else if (!(nodes.get(0) instanceof Element)) {
            DebugOut.assertFalse("xpath didn't return element", querystr);
        }
        return new XmlEltWrapper((Element) nodes.get(0));
    }

    public int int_attr(String name) {
        return Integer.parseInt(getAttributeValue(name));
    }

    public float float_attr(String name) {
        return Float.parseFloat(getAttributeValue(name));
    }
}
