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

    public Function getFunction(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("FcnDef")) {
            return getFunctionFromFcnDef(node);
        } else {
            throw new RuntimeException("no way to return a Function from a node of type " + typ);
        }
    }

    public StmtAssert getStmtAssert(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("SKAssertCall")) {
            return getStmtAssertFromSKAssertCall(node);
        } else {
            throw new RuntimeException("no way to return a StmtAssert from a node of type " + typ);
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

    public StreamSpec getStreamSpec(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("PackageDef")) {
            return getStreamSpecFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a StreamSpec from a node of type " + typ);
        }
    }

    public StmtVarDecl getStmtVarDecl(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("ValDef")) {
            return getStmtVarDeclFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a StmtVarDecl from a node of type " + typ);
        }
    }

    public Program getProgram(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("PackageDef")) {
            return getProgramFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a Program from a node of type " + typ);
        }
    }

    public Parameter getParameter(final GXLNode node) {
        String typ = GxlImport.nodeType(node);

        if (typ.equals("ValDef")) {
            return getParameterFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a Parameter from a node of type " + typ);
        }
    }



    // === Get a specific java type from a known GXL node type ===

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Function getFunctionFromFcnDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg2_tmp1 = followEdge("FcnDefSymbol", node); // gen marker 3
        String arg2 = getStringAttribute("symbolName", arg2_tmp1); // gen marker 7

        Type arg3 = getType(followEdge("FcnDefReturnTypeSymbol", node)); // gen marker 2

        Vector<Parameter> arg4_vec = new Vector<Parameter>();
        for (GXLNode arg4_tmp1 : followEdgeOL("FcnDefParamsList", node)) {
            arg4_vec.add(getParameter(arg4_tmp1)); // gen marker 4
        }
        List<Parameter> arg4 = unmodifiableList(arg4_vec);

        Statement arg5 = getStatement(followEdge("FcnBody", node)); // gen marker 2

        return new Function(arg0, Function.FUNC_WORK, arg2, arg3, arg4, arg5);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtAssert getStmtAssertFromSKAssertCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg1 = getExpression(followEdge("FcnArgList", node)); // gen marker 2

        return new StmtAssert(arg0, arg1, false);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypeStruct getTypeStructFromClassDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        String arg1 = getString(followEdge("ClassDefSymbol", node)); // gen marker 2

        Vector<String> arg2_vec = new Vector<String>();
        for (GXLNode arg2_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            arg2_vec.add(getStringAttribute("symbolName", arg2_tmp1)); // gen marker 6
        }
        List<String> arg2 = unmodifiableList(arg2_vec);

        Vector<Type> arg3_vec = new Vector<Type>();
        for (GXLNode arg3_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            GXLNode arg3_tmp2 = followEdge("TypeSymbol", arg3_tmp1); // gen marker 3
            arg3_vec.add(getType(followEdge("SketchType", arg3_tmp2))); // gen marker 1
        }
        List<Type> arg3 = unmodifiableList(arg3_vec);

        return new TypeStruct(arg0, arg1, arg2, arg3);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StreamSpec getStreamSpecFromPackageDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Vector<StmtVarDecl> arg5_vec = new Vector<StmtVarDecl>();
        for (GXLNode arg5_tmp1 : followEdgeUL("PackageDefGlobal", node)) {
            arg5_vec.add(getStmtVarDecl(arg5_tmp1)); // gen marker 4
        }
        List<StmtVarDecl> arg5 = unmodifiableList(arg5_vec);

        Vector<Function> arg6_vec = new Vector<Function>();
        for (GXLNode arg6_tmp1 : followEdgeUL("PackageDefFcn", node)) {
            arg6_vec.add(getFunction(arg6_tmp1)); // gen marker 4
        }
        List<Function> arg6 = unmodifiableList(arg6_vec);

        return new StreamSpec(arg0, StreamSpec.STREAM_FILTER, new StreamType((FEContext)null,
            TypePrimitive.bittype, TypePrimitive.bittype), "MAIN", Collections.EMPTY_LIST, arg5, arg6);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtVarDecl getStmtVarDeclFromValDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        Type arg1 = getType(followEdge("TypeSymbol", arg1_tmp1)); // gen marker 2

        GXLNode arg2_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg2 = getStringAttribute("symbolName", arg2_tmp1); // gen marker 7

        return new StmtVarDecl(arg0, arg1, arg2, null);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Program getProgramFromPackageDef(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        List<StreamSpec> arg1 = createSingleton(getStreamSpec(node)); // gen marker 8

        Vector<TypeStruct> arg2_vec = new Vector<TypeStruct>();
        for (GXLNode arg2_tmp1 : followEdgeUL("PackageDefElement", node)) {
            arg2_vec.add(getTypeStruct(arg2_tmp1)); // gen marker 4
        }
        List<TypeStruct> arg2 = unmodifiableList(arg2_vec);

        return new Program(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Parameter getParameterFromValDef(final GXLNode node) {

        GXLNode arg0_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        Type arg0 = getType(followEdge("TypeSymbol", arg0_tmp1)); // gen marker 2

        GXLNode arg1_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg1 = getStringAttribute("symbolName", arg1_tmp1); // gen marker 7

        return new Parameter(arg0, arg1);
    }



    // === Get by Java superclass ===

    public Type getType(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ClassDef")) {
            return getTypeStruct(node);
        } else {
            throw new RuntimeException("no way to return a Type from a node of type " + typ);
        }
    }

    public Expression getExpression(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        throw new RuntimeException("no gxl nodes corresponding to \"Expression\"; gxl type " + typ);
    }

    public Statement getStatement(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKAssertCall")) {
            return getStmtAssert(node);
        } if (typ.equals("ValDef")) {
            return getStmtVarDecl(node);
        } else {
            throw new RuntimeException("no way to return a Statement from a node of type " + typ);
        }
    }
}
