package sketch.util.wrapper;

import java.util.Vector;

import nu.xom.Element;
import nu.xom.Node;
import nu.xom.Nodes;

import org.w3c.dom.CDATASection;

import sketch.util.DebugOut;

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
        if (nodes.size() == 0) {
            throw new XmlNoXpathMatchException();
        } else if (nodes.size() > 1) {
            DebugOut
                    .assertFalse("multiple nodes returned from query", querystr);
        } else if (!(nodes.get(0) instanceof Element)) {
            DebugOut.assertFalse("xpath didn't return element", querystr);
        }
        return new XmlEltWrapper((Element) nodes.get(0));
    }

    public Vector<XmlEltWrapper> getChildXpathElements(String querystr) {
        Nodes nodes = query(querystr);
        Vector<XmlEltWrapper> elements = new Vector<XmlEltWrapper>();
        for (int a = 0; a < nodes.size(); a++) {
            Node node = nodes.get(a);
            if (node instanceof Element) {
                elements.add(new XmlEltWrapper((Element) node));
            }
        }
        return elements;
    }

    public Vector<String> getCDATAChildren() {
        Vector<String> result = new Vector<String>();
        int nchildren = getChildCount();
        for (int a = 0; a < nchildren; a++) {
            Node node = getChild(a);
            if (node instanceof CDATASection) {
                CDATASection cdata_node = (CDATASection) node;
                result.add(cdata_node.getData());
            }
        }
        return result;
    }

    public int int_attr(String name) {
        return Integer.parseInt(getAttributeValue(name));
    }

    public float float_attr(String name) {
        return Float.parseFloat(getAttributeValue(name));
    }
}
