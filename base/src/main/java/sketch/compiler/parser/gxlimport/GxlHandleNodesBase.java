package sketch.compiler.parser.gxlimport;

import net.sourceforge.gxl.GXLBool;
import net.sourceforge.gxl.GXLEdge;
import net.sourceforge.gxl.GXLInt;
import net.sourceforge.gxl.GXLNode;
import net.sourceforge.gxl.GXLString;
import scala.Tuple2;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Vector;

import sketch.compiler.ast.core.FEContext;
import sketch.compiler.ast.core.Function;
import sketch.compiler.ast.core.Parameter;
import sketch.compiler.ast.core.exprs.ExprNullPtr;
import sketch.compiler.ast.core.stmts.Statement;
import sketch.compiler.ast.core.typs.Type;
import sketch.util.DebugOut;
import sketch.util.datastructures.TypedHashMap;

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
    /** use to ensure all nodes are visited; stored in tuples of (GXLNode, java type) */
    public HashSet<Tuple2<GXLNode, String>> visited_nodes =
            new HashSet<Tuple2<GXLNode, String>>();
    public HashSet<GXLNode> visited_simple = new HashSet<GXLNode>();
    /**
     * certain string will cause Sketch to fail. GrGen already assigns unique names, so
     * character avoiding is all that's necessary.
     */
    protected TypedHashMap<String, String> names = new TypedHashMap<String, String>();

    public GxlHandleNodesBase(final GxlImport imprt) {
        this.imprt = imprt;
    }

    public void visit(final GXLNode node, final String java_name) {
        this.visited_nodes.add(new Tuple2<GXLNode, String>(node, java_name));
        this.visited_simple.add(node);
        DebugOut.assertSlow(this.visited_nodes.contains(new Tuple2<GXLNode, String>(node,
                java_name)));
    }

    public boolean hasEdge(final String name, final GXLNode node) {
        NodeStringTuple srckey = new NodeStringTuple(node, name);
        return !(this.imprt.edges_by_source.get(srckey).isEmpty());
    }

    public GXLNode followEdge(final String name, final GXLNode node) {
        NodeStringTuple srckey = new NodeStringTuple(node, name);
        Vector<GXLEdge> edges = this.imprt.edges_by_source.get(srckey);
        if (edges.size() != 1) {
            this.imprt.debugPrintNode("source node", node);
            for (GXLEdge edge : edges) {
                this.imprt.debugPrintEdge("edge", edge);
                DebugOut.print(edge.getSourceID(),
                        GxlImport.nodeType((GXLNode) edge.getSource()));
            }
            DebugOut.assertFalse("expected one edge of type", name, "from node",
                    GxlImport.nodeType(node), edges);
        }
        return (GXLNode) edges.get(0).getTarget();
    }

    public Vector<GXLNode> followEdgeOL(final String edge_name, final GXLNode node) {
        GXLNode lst = this.followEdge(edge_name, node);
        Vector<GXLNode> result = new Vector<GXLNode>();
        GXLNode lst_node = this.followEdge("ListNext", this.followEdge("ListFirst", lst));
        while (!GxlImport.nodeType(lst_node).equals("ListLastNode")) {
            result.add(this.followEdge("ListValue", lst_node));
            lst_node = this.followEdge("ListNext", lst_node);
        }
        if (result.isEmpty()) {
            DebugOut.fmt("List from edge '%s' from node of type '%s' is empty.",
                    edge_name, GxlImport.nodeType(node));
        }
        return result;
    }

    public Vector<GXLNode> followEdgeUL(final String edge_name, final GXLNode node) {
        NodeStringTuple srckey = new NodeStringTuple(node, edge_name);
        Vector<GXLEdge> edges = this.imprt.edges_by_source.get(srckey);
        Vector<GXLNode> nodes = new Vector<GXLNode>();
        for (GXLEdge edge : edges) {
            nodes.add((GXLNode) edge.getTarget());
        }
        // string sorted -- only for consistent order
        Collections.sort(nodes, new Comparator<GXLNode>() {
            public int compare(final GXLNode o1, final GXLNode o2) {
                return o1.getID().compareTo(o2.getID());
            }
        });
        if (nodes.isEmpty()) {
            DebugOut.assertSlow(edges.isEmpty(), "edges not empty but nodes empty.");
            DebugOut.fmt("UL from edge '%s' from node of type '%s' is empty.", edge_name,
                    GxlImport.nodeType(node));
        }
        return nodes;
    }

    public String getStringAttribute(final String name, final GXLNode node) {
        return ((GXLString) node.getAttr(name).getValue()).getValue();
    }

    public boolean getBooleanAttribute(final String name, final GXLNode node) {
        return ((GXLBool) node.getAttr(name).getValue()).getBooleanValue();
    }

    public int getIntAttribute(final String name, final GXLNode node) {
        return ((GXLInt) node.getAttr(name).getValue()).getIntValue();
    }

    public <V> List<V> createSingleton(final V v) {
        Vector<V> tmp = new Vector<V>(1);
        tmp.add(v);
        return Collections.unmodifiableList(tmp);
    }

    public FEContext create_fe_context(final GXLNode node) {
        String srcfile = this.getStringAttribute("sourceFile", node);
        int line = this.getIntAttribute("startLine", node);
        int col = this.getIntAttribute("startCol", node);
        return new FEContext(srcfile, line, col);
    }

    public String getImplements(final GXLNode node) {
        if (this.hasEdge("SKImplements", node)) {
            GXLNode fcndef = this.followEdge("SKImplements", node);
            GXLNode sym = this.followEdge("FcnDefSymbol", fcndef);
            return this.getStringAttribute("name", this.followEdge("PrintSymName", sym));
        }
        return null;
    }

    @SuppressWarnings("deprecation")
    public Function createFunction(final FEContext arg0, final String arg1,
            final Type arg2, final List<Parameter> arg3, final String arg4,
            final Statement arg5, final boolean isGenerator)
    {
        if (isGenerator) {
            return Function.newHelper(arg0, arg1, arg2, arg3, arg4, arg5);
        } else {
            return Function.newStatic(arg0, arg1, arg2, arg3, arg4, arg5);
        }
    }

    protected ExprNullPtr createExprNullPtr(final FEContext arg0) {
        return ExprNullPtr.nullPtr;
    }

    protected String createString(final String arg0) {
        final String existing = this.names.get(arg0);
        if (existing != null) {
            return existing;
        } else {
            String next =
                    arg0.replaceAll("\\$", "").replaceAll("<", "LT_").replaceAll(">",
                            "_GT");
            // very unlikely
            while (this.names.containsValue(next)) {
                next += "_";
            }
            this.names.put(arg0, next);
            // DebugOut.print("sanitized string", next);
            return next;
        }
    }

    protected <T> T getSingleton(List<T> lst) {
        if (lst.size() != 1) {
            DebugOut.assertFalse("getSingleton() error -- list has " + lst.size() +
                    " elements");
        }
        return lst.get(0);
    }
}
