package sketch.compiler.parser.gxlimport;

import java.util.Collections;
import java.util.Comparator;
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
 * @copyright University of California, Berkeley 2009
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GxlHandleNodes extends GxlHandleNodesBase {
    public GxlHandleNodes(final GxlImport imprt) {
        super(imprt);
    }



    // === Get a specific java type, branching on the current GXL node type ===

    public Program getProgram(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("PackageDef")) {
            return getProgramFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a Program from a node of type " + typ);
        }
    }

    public StreamSpec getStreamSpec(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("PackageDef")) {
            return getStreamSpecFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a StreamSpec from a node of type " + typ);
        }
    }

    public TypeStruct getTypeStruct(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("ClassDef")) {
            return getTypeStructFromClassDef(node);
        } else {
            throw new RuntimeException("no way to return a TypeStruct from a node of type " + typ);
        }
    }



    // === Get a specific java type from a known GXL node type ===

    public Program getProgramFromPackageDef(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        StreamSpec arg1 = getStreamSpec(node);

        Vector<TypeStruct> arg2_vec = new Vector<TypeStruct>();
        for (GXLNode arg2_tmp1 : followEdgeUL("PackageDefElement", node)) {
            arg2_vec.add(getTypeStruct(arg2_tmp1));
        }
        List<TypeStruct> arg2 = unmodifiableList(arg2_vec);

        return new Program(arg0, arg1, arg2);
    }

    public StreamSpec getStreamSpecFromPackageDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Vector<StmtVarDecl> arg1_vec = new Vector<StmtVarDecl>();
        for (GXLNode arg1_tmp1 : followEdgeUL("PackageDefGlobal", node)) {
            arg1_vec.add(getStmtVarDecl(arg1_tmp1));
        }
        List<StmtVarDecl> arg1 = unmodifiableList(arg1_vec);

        Vector<Function> arg2_vec = new Vector<Function>();
        for (GXLNode arg2_tmp1 : followEdgeUL("PackageDefElement", node)) {
            arg2_vec.add(getFunction(arg2_tmp1));
        }
        List<Function> arg2 = unmodifiableList(arg2_vec);

        return createStreamSpec(arg0, arg1, arg2);
    }

    public TypeStruct getTypeStructFromClassDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        String arg1 = getString(followEdge("ClassDefSymbol", node));

        Vector<String> arg2_vec = new Vector<String>();
        for (GXLNode arg2_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            arg2_vec.add(getStringAttribute("symbolName", arg2_tmp1));
        }
        List<String> arg2 = unmodifiableList(arg2_vec);

        Vector<Type> arg3_vec = new Vector<Type>();
        for (GXLNode arg3_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            GXLNode arg3_tmp2 = followEdge("TypeSymbol", arg3_tmp1);
            arg3_vec.add(getType(followEdge("SketchType", arg3_tmp2)));
        }
        List<Type> arg3 = unmodifiableList(arg3_vec);

        return new TypeStruct(arg0, arg1, arg2, arg3);
    }



    // === Get by Java superclass ===

    public Type getType(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        throw new RuntimeException("no gxl nodes corresponding to \"Type\"");
    }
}
