package sketch.compiler.parser.gxlimport;

import java.util.List;
import java.util.Vector;
import static java.util.Collections.unmodifiableList;
import net.sourceforge.gxl.*;
import sketch.compiler.ast.core.*;
import sketch.compiler.ast.core.exprs.*;
import sketch.compiler.ast.core.stmts.*;
import sketch.compiler.ast.core.typs.*;



/**
 * Handle simple node types. THIS IS A GENERATED FILE, modify the .jinja2 version.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GxlHandleNodes {
    public GxlImport imprt;

    public GxlHandleNodes(final GxlImport imprt) {
        this.imprt = imprt;
    }

    public GXLNode followEdge(final String name, final GXLNode node) {
        return null;
    }

    public String getString(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        throw new RuntimeException("getString not implemented for " + typ);
    }

    public Vector<GXLNode> followEdgeList(final String edge_name, final GXLNode node) {
        GXLNode lst = this.followEdge(edge_name, node);
        Vector<GXLNode> result = new Vector<GXLNode>();
        GXLNode lst_node = this.followEdge("ListNext", this.followEdge("ListFirst", lst));
        while (!GxlImport.nodeType(lst_node).equals("ListLastNode")) {
            result.add(this.followEdge("ListValue", lst_node));
            lst_node = this.followEdge("ListNext", lst_node);
        }
        return result;
    }

    public String getStringAttribute(final String name, final GXLNode node) {
        return ((GXLString) node.getAttr(name).getValue()).getValue();
    }

    public TypeStruct getTypeStruct(GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ClassDef")) {
            FEContext arg0 = new FEContext();

            String arg1 = getString(followEdge("ClassDefSymbol", node));


            Vector<String> arg2_vec = new Vector<String>();
            for (GXLNode arg2_tmp1 : followEdgeList("ClassDefFieldsList", node)) {
                arg2_vec.add(getStringAttribute("symbolName", arg2_tmp1));
            }
            List<String> arg2 = unmodifiableList(arg2_vec);


            Vector<Type> arg3_vec = new Vector<Type>();
            for (GXLNode arg3_tmp1 : followEdgeList("ClassDefFieldsList", node)) {
                GXLNode arg3_tmp2 = followEdge("TypeSymbol", arg3_tmp1);
                arg3_vec.add(getType(followEdge("SketchType", arg3_tmp2)));
            }
            List<Type> arg3 = unmodifiableList(arg3_vec);

            return new TypeStruct(arg0, arg1, arg2, arg3);
        } else {
            throw new RuntimeException("no way to return a TypeStruct from a node of type " + typ);
        }
    }

    public Type getType(GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ClassDef")) {
            return getTypeStruct(node);
        } else {
            throw new RuntimeException("no way to return a Type from a node of type " + typ);
        }
    }

}
