package sketch.compiler.parser.gxlimport;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

import net.sourceforge.gxl.GXLEdge;
import net.sourceforge.gxl.GXLInt;
import net.sourceforge.gxl.GXLNode;
import net.sourceforge.gxl.GXLString;
import sketch.compiler.ast.core.FEContext;
import sketch.compiler.ast.core.Function;
import sketch.compiler.ast.core.StreamSpec;
import sketch.compiler.ast.core.stmts.StmtVarDecl;

/**
 * Handle simple node types. Non-generated functions; generated functions are added in the
 * subclass GxlHandleNodes.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class GxlHandleNodesBase {
    public GxlImport imprt;

    public GxlHandleNodesBase(final GxlImport imprt) {
        this.imprt = imprt;
    }

    public GXLNode followEdge(final String name, final GXLNode node) {
        Vector<GXLEdge> edges = this.imprt.edges_by_source_id.get(node);
        assert edges.size() == 1;
        return this.imprt.nodes_by_id.get(edges.get(0).getTargetID());
    }

    public String getString(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        throw new RuntimeException("getString not implemented for " + typ);
    }

    public Vector<GXLNode> followEdgeOL(final String edge_name, final GXLNode node) {
        GXLNode lst = this.followEdge(edge_name, node);
        Vector<GXLNode> result = new Vector<GXLNode>();
        GXLNode lst_node = this.followEdge("ListNext", this.followEdge("ListFirst", lst));
        while (!GxlImport.nodeType(lst_node).equals("ListLastNode")) {
            result.add(this.followEdge("ListValue", lst_node));
            lst_node = this.followEdge("ListNext", lst_node);
        }
        return result;
    }

    public Vector<GXLNode> followEdgeUL(final String edge_name, final GXLNode node) {
        Vector<GXLEdge> edges = this.imprt.edges_by_source_id.get(node);
        Vector<GXLNode> nodes = new Vector<GXLNode>();
        for (GXLEdge edge : edges) {
            nodes.add(this.imprt.nodes_by_id.get(edge.getTargetID()));
        }
        // string sorted -- only for consistent order
        Collections.sort(nodes, new Comparator<GXLNode>() {
            public int compare(final GXLNode o1, final GXLNode o2) {
                return o1.getID().compareTo(o2.getID());
            }
        });
        return nodes;
    }

    public String getStringAttribute(final String name, final GXLNode node) {
        return ((GXLString) node.getAttr(name).getValue()).getValue();
    }

    public int getIntAttribute(final String name, final GXLNode node) {
        return ((GXLInt) node.getAttr(name).getValue()).getIntValue();
    }

    public FEContext create_fe_context(final GXLNode node) {
        String srcfile = this.getStringAttribute("sourceFile", node);
        int line = this.getIntAttribute("startLine", node);
        int col = this.getIntAttribute("startCol", node);
        return new FEContext(srcfile, line, col);
    }

    public StreamSpec createStreamSpec(final FEContext ctx,
            final List<StmtVarDecl> glbls, final List<Function> fcns) {
        // return new StreamSpec(ctx, StreamSpec.STREAM_FILTER, st, name, params, vars,
        // funcs)
        return null;
    }
}
