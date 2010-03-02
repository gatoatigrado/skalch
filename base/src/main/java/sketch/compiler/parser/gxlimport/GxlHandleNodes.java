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
import sketch.compiler.ast.scala.exprs.*;

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



    // === Get a specific java type from a known GXL node type ===

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
    public TypeStruct getTypeStructFromClassDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("ClassDefSymbol", node); // gen marker 3
        GXLNode arg1_tmp2 = followEdge("PrintSymName", arg1_tmp1); // gen marker 3
        String arg1 = getStringAttribute("name", arg1_tmp2); // gen marker 7

        Vector<String> arg2_vec = new Vector<String>();
        for (GXLNode arg2_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            GXLNode arg2_tmp2 = followEdge("PrintSymName", arg2_tmp1); // gen marker 3
            arg2_vec.add(getStringAttribute("name", arg2_tmp2)); // gen marker 6
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
    public Function getFunctionFromFcnDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("FcnDefSymbol", node); // gen marker 3
        String arg1 = getStringAttribute("symbolName", arg1_tmp1); // gen marker 7

        Type arg2 = getType(followEdge("FcnDefReturnTypeSymbol", node)); // gen marker 2

        Vector<Parameter> arg3_vec = new Vector<Parameter>();
        for (GXLNode arg3_tmp1 : followEdgeOL("FcnDefParamsList", node)) {
            arg3_vec.add(getParameter(arg3_tmp1)); // gen marker 4
        }
        List<Parameter> arg3 = unmodifiableList(arg3_vec);

        Statement arg5 = getStatement(followEdge("FcnBody", node)); // gen marker 2

        return createFunction(arg0, arg1, arg2, arg3, getImplements(node), arg5);
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
    public Parameter getParameterFromValDef(final GXLNode node) {
        GXLNode arg0_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        Type arg0 = getType(followEdge("TypeSymbol", arg0_tmp1)); // gen marker 2

        GXLNode arg1_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg1 = getStringAttribute("symbolName", arg1_tmp1); // gen marker 7

        return new Parameter(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprVar getExprVarFromVarRef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("VarRefSymbol", node); // gen marker 3
        String arg1 = getStringAttribute("symbolName", arg1_tmp1); // gen marker 7

        return new ExprVar(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtAssert getStmtAssertFromSKAssertCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg1 = getExpression(followEdge("SKAssertCallArg", node)); // gen marker 2

        return new StmtAssert(arg0, arg1, false);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtBlock getStmtBlockFromSKBlock(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Vector<Statement> arg1_vec = new Vector<Statement>();
        for (GXLNode arg1_tmp1 : followEdgeOL("BlockStmtList", node)) {
            arg1_vec.add(getStatement(arg1_tmp1)); // gen marker 4
        }
        List<Statement> arg1 = unmodifiableList(arg1_vec);

        return new StmtBlock(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtReturn getStmtReturnFromReturn(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg1 = getExpression(followEdge("ReturnExpr", node)); // gen marker 2

        return new StmtReturn(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtAssign getStmtAssignFromAssign(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Expression arg1 = getExpression(followEdge("AssignLhs", node)); // gen marker 2

        Expression arg2 = getExpression(followEdge("AssignRhs", node)); // gen marker 2

        return new StmtAssign(arg0, arg1, arg2, 0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtIfThen getStmtIfThenFromIf(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg1 = getExpression(followEdge("IfCond", node)); // gen marker 2

        Statement arg2 = getStatement(followEdge("IfThen", node)); // gen marker 2

        Statement arg3 = getStatement(followEdge("IfElse", node)); // gen marker 2

        return new StmtIfThen(arg0, arg1, arg2, arg3);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprBinary getExprBinaryFromFcnBinaryCall(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Expression arg1 = getExpression(followEdge("FcnBinaryCallLhs", node)); // gen marker 2

        String arg2 = getStringAttribute("strop", node); // gen marker 7

        Expression arg3 = getExpression(followEdge("FcnBinaryCallRhs", node)); // gen marker 2

        return new ExprBinary(arg0, arg1, arg2, arg3);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprUnary getExprUnaryFromFcnCallUnaryNegative(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg2 = getExpression(followEdge("FcnArgChain", node)); // gen marker 2

        return new ExprUnary(arg0, ExprUnary.UNOP_NEG, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprStar getExprStarFromHoleCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        return new ExprStar(arg0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprConstInt getExprConstIntFromIntConstant(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        int arg1 = getIntAttribute("value", node); // gen marker 7

        return new ExprConstInt(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprConstUnit getExprConstUnitFromUnitConstant(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        return new ExprConstUnit(arg0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypePrimitive getTypePrimitiveFromTypeBoolean(final GXLNode node) {
        return TypePrimitive.bittype;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypePrimitive getTypePrimitiveFromTypeInt(final GXLNode node) {
        return TypePrimitive.inttype;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypePrimitive getTypePrimitiveFromTypeUnit(final GXLNode node) {
        return TypePrimitive.voidtype;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Type getTypeFromSymbol(final GXLNode node) {
        return getType(followEdge("SketchType", node));
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypeStructRef getTypeStructRefFromTypeStructRef(final GXLNode node) {
        String arg0 = getStringAttribute("typename", node); // gen marker 7

        return new TypeStructRef(arg0);
    }



    // === Get by Java superclass ===

    public StmtReturn getStmtReturn(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("Return")) {
            return getStmtReturnFromReturn(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtReturn' from a node of type " + typ);
        }
    }

    public Statement getStatement(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("If")) {
            return getStmtIfThenFromIf(node);
        } else if (typ.equals("Assign")) {
            return getStmtAssignFromAssign(node);
        } else if (typ.equals("Return")) {
            return getStmtReturnFromReturn(node);
        } else if (typ.equals("SKBlock")) {
            return getStmtBlockFromSKBlock(node);
        } else if (typ.equals("SKAssertCall")) {
            return getStmtAssertFromSKAssertCall(node);
        } else if (typ.equals("ValDef")) {
            return getStmtVarDeclFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a 'Statement' from a node of type " + typ);
        }
    }

    public ExprUnary getExprUnary(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("FcnCallUnaryNegative")) {
            return getExprUnaryFromFcnCallUnaryNegative(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprUnary' from a node of type " + typ);
        }
    }

    public ExprConstUnit getExprConstUnit(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprConstUnit' from a node of type " + typ);
        }
    }

    public Program getProgram(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("PackageDef")) {
            return getProgramFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a 'Program' from a node of type " + typ);
        }
    }

    public ExprBinary getExprBinary(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("FcnBinaryCall")) {
            return getExprBinaryFromFcnBinaryCall(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprBinary' from a node of type " + typ);
        }
    }

    public StmtAssert getStmtAssert(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKAssertCall")) {
            return getStmtAssertFromSKAssertCall(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtAssert' from a node of type " + typ);
        }
    }

    public Type getType(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("TypeStructRef")) {
            return getTypeStructRefFromTypeStructRef(node);
        } else if (typ.equals("TypeUnit")) {
            return getTypePrimitiveFromTypeUnit(node);
        } else if (typ.equals("TypeInt")) {
            return getTypePrimitiveFromTypeInt(node);
        } else if (typ.equals("TypeBoolean")) {
            return getTypePrimitiveFromTypeBoolean(node);
        } else if (typ.equals("ClassDef")) {
            return getTypeStructFromClassDef(node);
        } else if (typ.equals("Symbol")) {
            return getTypeFromSymbol(node);
        } else {
            throw new RuntimeException("no way to return a 'Type' from a node of type " + typ);
        }
    }

    public TypePrimitive getTypePrimitive(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("TypeUnit")) {
            return getTypePrimitiveFromTypeUnit(node);
        } else if (typ.equals("TypeInt")) {
            return getTypePrimitiveFromTypeInt(node);
        } else if (typ.equals("TypeBoolean")) {
            return getTypePrimitiveFromTypeBoolean(node);
        } else {
            throw new RuntimeException("no way to return a 'TypePrimitive' from a node of type " + typ);
        }
    }

    public Function getFunction(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("FcnDef")) {
            return getFunctionFromFcnDef(node);
        } else {
            throw new RuntimeException("no way to return a 'Function' from a node of type " + typ);
        }
    }

    public ExprVar getExprVar(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("VarRef")) {
            return getExprVarFromVarRef(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprVar' from a node of type " + typ);
        }
    }

    public ExprConstInt getExprConstInt(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprConstInt' from a node of type " + typ);
        }
    }

    public StmtAssign getStmtAssign(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("Assign")) {
            return getStmtAssignFromAssign(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtAssign' from a node of type " + typ);
        }
    }

    public Parameter getParameter(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ValDef")) {
            return getParameterFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a 'Parameter' from a node of type " + typ);
        }
    }

    public StreamSpec getStreamSpec(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("PackageDef")) {
            return getStreamSpecFromPackageDef(node);
        } else {
            throw new RuntimeException("no way to return a 'StreamSpec' from a node of type " + typ);
        }
    }

    public StmtVarDecl getStmtVarDecl(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ValDef")) {
            return getStmtVarDeclFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtVarDecl' from a node of type " + typ);
        }
    }

    public ExprConstant getExprConstant(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprConstant' from a node of type " + typ);
        }
    }

    public TypeStruct getTypeStruct(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ClassDef")) {
            return getTypeStructFromClassDef(node);
        } else {
            throw new RuntimeException("no way to return a 'TypeStruct' from a node of type " + typ);
        }
    }

    public StmtBlock getStmtBlock(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKBlock")) {
            return getStmtBlockFromSKBlock(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtBlock' from a node of type " + typ);
        }
    }

    public TypeStructRef getTypeStructRef(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("TypeStructRef")) {
            return getTypeStructRefFromTypeStructRef(node);
        } else {
            throw new RuntimeException("no way to return a 'TypeStructRef' from a node of type " + typ);
        }
    }

    public Expression getExpression(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else if (typ.equals("FcnCallUnaryNegative")) {
            return getExprUnaryFromFcnCallUnaryNegative(node);
        } else if (typ.equals("VarRef")) {
            return getExprVarFromVarRef(node);
        } else if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else if (typ.equals("HoleCall")) {
            return getExprStarFromHoleCall(node);
        } else if (typ.equals("FcnBinaryCall")) {
            return getExprBinaryFromFcnBinaryCall(node);
        } else {
            throw new RuntimeException("no way to return a 'Expression' from a node of type " + typ);
        }
    }

    public StmtIfThen getStmtIfThen(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("If")) {
            return getStmtIfThenFromIf(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtIfThen' from a node of type " + typ);
        }
    }

    public ExprStar getExprStar(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("HoleCall")) {
            return getExprStarFromHoleCall(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprStar' from a node of type " + typ);
        }
    }
}
