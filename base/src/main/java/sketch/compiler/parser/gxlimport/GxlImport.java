package sketch.compiler.parser.gxlimport;

import java.io.File;
import java.util.HashMap;
import java.util.Vector;

import net.sourceforge.gxl.GXLDocument;
import net.sourceforge.gxl.GXLEdge;
import net.sourceforge.gxl.GXLGXL;
import net.sourceforge.gxl.GXLGraph;
import net.sourceforge.gxl.GXLGraphElement;
import net.sourceforge.gxl.GXLNode;
import sketch.compiler.ast.core.StreamSpec;
import sketch.compiler.ast.core.typs.TypeStruct;
import sketch.util.DefaultHashMap;

/**
 * Import all nodes from a GXL file and create a Program instance.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GxlImport {
    public DefaultHashMap<String, Vector<GXLNode>> nodes_by_type;
    public HashMap<String, GXLNode> nodes_by_id;
    public DefaultHashMap<String, Vector<GXLEdge>> edges_by_source_id;
    public DefaultHashMap<String, Vector<GXLEdge>> edges_by_target_id;
    public Vector<TypeStruct> structs;
    public Vector<StreamSpec> streams;

    protected void init(final GXLGraph graph) {
        this.structs = new Vector<TypeStruct>();
        this.streams = new Vector<StreamSpec>();
        this.nodes_by_type = new DefaultHashMap<String, Vector<GXLNode>>();
        this.nodes_by_type.defvalue = this.nodes_by_type.new DefValueGenerator() {
            @Override
            public Vector<GXLNode> get_value() {
                return new Vector<GXLNode>();
            }
        };
        this.nodes_by_id = new HashMap<String, GXLNode>();
        this.edges_by_source_id = new DefaultHashMap<String, Vector<GXLEdge>>();
        this.edges_by_target_id = new DefaultHashMap<String, Vector<GXLEdge>>();
        this.edges_by_source_id.defvalue =
            this.edges_by_target_id.defvalue =
                this.edges_by_target_id.new DefValueGenerator() {
            @Override
            public Vector<GXLEdge> get_value() {
                return new Vector<GXLEdge>();
            }
        };
        for (int a = 0; a < graph.getGraphElementCount(); a++) {
            GXLGraphElement elt = graph.getGraphElementAt(a);
            if (elt instanceof GXLNode) {
                GXLNode elt2 = (GXLNode) elt;
                this.nodes_by_type.get(GxlImport.nodeType(elt2)).add(elt2);
                this.nodes_by_id.put(elt2.getID(), elt2);
            } else if (elt instanceof GXLEdge) {
                GXLEdge elt2 = (GXLEdge) elt;
                this.edges_by_source_id.get(elt2.getSourceID()).add(elt2);
                this.edges_by_target_id.get(elt2.getTargetID()).add(elt2);
            }
        }
        for (GXLNode pkg : this.nodes_by_type.get("PackageDef")) {
            this.handleNode(pkg);
        }
    }

    private void handleNode(final GXLNode pkg) {
    }

    public GxlImport(final GXLGraph graph) {
        this.init(graph);
    }

    public GxlImport(final File graph_file) {
        GXLDocument doc = null;
        try {
            doc = new GXLDocument(graph_file);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        GXLGXL gxl = doc.getDocumentElement();
        for (int i = 0; i < gxl.getGraphCount(); i++) {
            GXLGraph graph = gxl.getGraphAt(i);
            if (graph.getID() == "DefaultGraph") {
                this.init(graph);
                return;
            }
        }
        assert false : "couldn't find a graph named 'DefaultGraph'";
    }

    public static String nodeType(final GXLNode node) {
        String uri = node.getType().getURI().toString();
        assert uri.startsWith("#") : "node type not relative " + uri;
        return uri.substring(1);
    }
}
