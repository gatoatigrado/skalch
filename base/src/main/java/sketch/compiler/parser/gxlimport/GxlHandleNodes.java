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
import sketch.compiler.ast.cuda.exprs.*;
import sketch.compiler.ast.cuda.stmts.*;
import sketch.compiler.ast.core.typs.*;
import sketch.compiler.ast.scala.exprs.*;
import sketch.compiler.ast.cuda.typs.CudaMemoryType;
import sketch.util.datastructures.TprintTuple;

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

        Vector<FieldDecl> arg5_vec = new Vector<FieldDecl>();
        for (GXLNode arg5_tmp1 : followEdgeUL("PackageDefGlobal", node)) {
            arg5_vec.add(getFieldDecl(arg5_tmp1)); // gen marker 4
        }
        List<FieldDecl> arg5 = unmodifiableList(arg5_vec);

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
        String arg1 = getString(followEdge("PrintSymName", arg1_tmp1)); // gen marker 2

        Vector<String> arg2_vec = new Vector<String>();
        for (GXLNode arg2_tmp1 : followEdgeOL("ClassDefFieldsList", node)) {
            arg2_vec.add(getString(followEdge("PrintSymName", arg2_tmp1))); // gen marker 1
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
        String arg1 = getString(followEdge("PrintSymName", arg1_tmp1)); // gen marker 2

        Type arg2 = getType(followEdge("FcnDefReturnTypeSymbol", node)); // gen marker 2

        Vector<Parameter> arg3_vec = new Vector<Parameter>();
        for (GXLNode arg3_tmp1 : followEdgeOL("FcnDefParamsList", node)) {
            arg3_vec.add(getParameter(arg3_tmp1)); // gen marker 4
        }
        List<Parameter> arg3 = unmodifiableList(arg3_vec);

        Statement arg5 = getStatement(followEdge("FcnBody", node)); // gen marker 2

        GXLNode arg6_tmp1 = followEdge("FcnDefIsGenerator", node); // gen marker 3
        boolean arg6 = getBooleanAttribute("value", arg6_tmp1); // gen marker 7

        return createFunction(arg0, arg1, arg2, arg3, getImplements(node), arg5, arg6);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtVarDecl getStmtVarDeclFromValDef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Type arg1 = getType(followEdge("ValDefSymbol", node)); // gen marker 2

        GXLNode arg2_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg2 = getString(followEdge("PrintSymName", arg2_tmp1)); // gen marker 2

        return new StmtVarDecl(arg0, arg1, arg2, null);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Parameter getParameterFromValDef(final GXLNode node) {
        Type arg0 = getType(followEdge("ValDefSymbol", node)); // gen marker 2

        GXLNode arg1_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg1 = getString(followEdge("PrintSymName", arg1_tmp1)); // gen marker 2

        GXLNode arg2_tmp1 = followEdge("SketchParamType", node); // gen marker 3
        int arg2 = getIntAttribute("typecode", arg2_tmp1); // gen marker 7

        return new Parameter(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public FieldDecl getFieldDeclFromValDef(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Type arg1 = getType(followEdge("ValDefSymbol", node)); // gen marker 2

        GXLNode arg2_tmp1 = followEdge("ValDefSymbol", node); // gen marker 3
        String arg2 = getString(followEdge("PrintSymName", arg2_tmp1)); // gen marker 2

        return new FieldDecl(arg0, arg1, arg2, null);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprVar getExprVarFromVarRef(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("VarRefSymbol", node); // gen marker 3
        String arg1 = getString(followEdge("PrintSymName", arg1_tmp1)); // gen marker 2

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
    public StmtExpr getStmtExprFromSKStmtExpr(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Expression arg1 = getExpression(followEdge("SKStmtExprExpr", node)); // gen marker 2

        return new StmtExpr(arg0, arg1);
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
    public StmtWhile getStmtWhileFromSKWhileLoop(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Expression arg1 = getExpression(followEdge("SKWhileLoopCond", node)); // gen marker 2

        Statement arg2 = getStatement(followEdge("SKWhileLoopBody", node)); // gen marker 2

        return new StmtWhile(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaSyncthreads getCudaSyncthreadsFromSyncthreadsCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        return new CudaSyncthreads(arg0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public StmtEmpty getStmtEmptyFromUnitConstant(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        return new StmtEmpty(arg0);
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

        Vector<Expression> arg2_vec = new Vector<Expression>();
        for (GXLNode arg2_tmp1 : followEdgeOL("FcnArgList", node)) {
            arg2_vec.add(getExpression(arg2_tmp1)); // gen marker 4
        }
        Expression arg2 = getSingleton(unmodifiableList(arg2_vec));

        return new ExprUnary(arg0, ExprUnary.UNOP_NEG, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprFunCall getExprFunCallFromFcnCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        GXLNode arg1_tmp1 = followEdge("FcnCallSymbol", node); // gen marker 3
        String arg1 = getString(followEdge("PrintSymName", arg1_tmp1)); // gen marker 2

        Vector<Expression> arg2_vec = new Vector<Expression>();
        for (GXLNode arg2_tmp1 : followEdgeOL("FcnArgList", node)) {
            arg2_vec.add(getExpression(arg2_tmp1)); // gen marker 4
        }
        List<Expression> arg2 = unmodifiableList(arg2_vec);

        return new ExprFunCall(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprNew getExprNewFromSKNew(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Type arg1 = getType(followEdge("FcnCallTypeSymbol", node)); // gen marker 2

        return new ExprNew(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprField getExprFieldFromFieldAccess(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Expression arg1 = getExpression(followEdge("FieldAccessObject", node)); // gen marker 2

        GXLNode arg2_tmp1 = followEdge("FieldAccessSymbol", node); // gen marker 3
        String arg2 = getString(followEdge("PrintSymName", arg2_tmp1)); // gen marker 2

        return new ExprField(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprStar getExprStarFromHoleCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Type arg2 = getType(followEdge("FcnCallTypeSymbol", node)); // gen marker 2

        return new ExprStar(arg0, 4, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprNullPtr getExprNullPtrFromNullTypeConstant(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        return new ExprNullPtr(arg0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprConstBoolean getExprConstBooleanFromBooleanConstant(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        boolean arg1 = getBooleanAttribute("value", node); // gen marker 7

        return new ExprConstBoolean(arg0, arg1);
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
    public ExprArrayInit getExprArrayInitFromNewArray(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Vector<Expression> arg1_vec = new Vector<Expression>();
        for (GXLNode arg1_tmp1 : followEdgeOL("ArrValueList", node)) {
            arg1_vec.add(getExpression(arg1_tmp1)); // gen marker 4
        }
        List<Expression> arg1 = unmodifiableList(arg1_vec);

        return new ExprArrayInit(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprArrayRange getExprArrayRangeFromSketchArrayAccess(final GXLNode node) {
        FENode arg0 = new DummyFENode(create_fe_context(node));

        Expression arg1 = getExpression(followEdge("SketchArrayAccessArray", node)); // gen marker 2

        Expression arg2 = getExpression(followEdge("SketchArrayAccessIndex", node)); // gen marker 2

        return new ExprArrayRange(arg0, arg1, arg2);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public ExprTprint getExprTprintFromSketchTprintCall(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        Vector<TprintTuple> arg1_vec = new Vector<TprintTuple>();
        for (GXLNode arg1_tmp1 : followEdgeOL("PrintCallArgList", node)) {
            arg1_vec.add(getTprintTuple(arg1_tmp1)); // gen marker 4
        }
        List<TprintTuple> arg1 = unmodifiableList(arg1_vec);

        return new ExprTprint(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TprintTuple getTprintTupleFromSketchPrintTuple(final GXLNode node) {
        GXLNode arg0_tmp1 = followEdge("SketchPrintTupleName", node); // gen marker 3
        String arg0 = getStringAttribute("value", arg0_tmp1); // gen marker 7

        Expression arg1 = getExpression(followEdge("SketchPrintTupleValue", node)); // gen marker 2

        return new TprintTuple(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaThreadIdx getCudaThreadIdxFromSketchThreadIdx(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        String arg1 = getStringAttribute("indexName", node); // gen marker 7

        return new CudaThreadIdx(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaBlockDim getCudaBlockDimFromSketchBlockDim(final GXLNode node) {
        FEContext arg0 = create_fe_context(node);

        String arg1 = getStringAttribute("indexName", node); // gen marker 7

        return new CudaBlockDim(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public Type getTypeFromSymbol(final GXLNode node) {
        Type arg0 = getType(followEdge("SketchType", node)); // gen marker 2

        CudaMemoryType arg1 = getCudaMemoryType(followEdge("TermMemLocationType", node)); // gen marker 2

        return createType(arg0, arg1);
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
    public TypeArray getTypeArrayFromTypeArray(final GXLNode node) {
        Type arg0 = getType(followEdge("ArrayInnerTypeSymbol", node)); // gen marker 2

        Expression arg1 = getExpression(followEdge("ArrayLengthExpr", node)); // gen marker 2

        return new TypeArray(arg0, arg1);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public TypeStructRef getTypeStructRefFromTypeStructRef(final GXLNode node) {
        return new TypeStructRef(createString(getStringAttribute("typename", node)));
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public String getStringFromPrintName(final GXLNode node) {
        String arg0 = getStringAttribute("name", node); // gen marker 7

        return createString(arg0);
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaMemoryType getCudaMemoryTypeFromCudaMemShared(final GXLNode node) {
        return CudaMemoryType.GLOBAL;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaMemoryType getCudaMemoryTypeFromCudaMemImplicitShared(final GXLNode node) {
        return CudaMemoryType.GLOBAL;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaMemoryType getCudaMemoryTypeFromCudaMemDefaultShared(final GXLNode node) {
        return CudaMemoryType.GLOBAL;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaMemoryType getCudaMemoryTypeFromCudaMemGlobal(final GXLNode node) {
        return CudaMemoryType.GLOBAL;
    }

    // NOTE -- some constructors are marked deprecated to avoid later use.
    @SuppressWarnings("deprecation")
    public CudaMemoryType getCudaMemoryTypeFromCudaMemLocal(final GXLNode node) {
        return CudaMemoryType.LOCAL;
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
        if (typ.equals("UnitConstant")) {
            return getStmtEmptyFromUnitConstant(node);
        } else if (typ.equals("SyncthreadsCall")) {
            return getCudaSyncthreadsFromSyncthreadsCall(node);
        } else if (typ.equals("SKWhileLoop")) {
            return getStmtWhileFromSKWhileLoop(node);
        } else if (typ.equals("SKStmtExpr")) {
            return getStmtExprFromSKStmtExpr(node);
        } else if (typ.equals("If")) {
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
        if (typ.equals("TypeArray")) {
            return getTypeArrayFromTypeArray(node);
        } else if (typ.equals("TypeStructRef")) {
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

    public TypeArray getTypeArray(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("TypeArray")) {
            return getTypeArrayFromTypeArray(node);
        } else {
            throw new RuntimeException("no way to return a 'TypeArray' from a node of type " + typ);
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

    public CudaBlockDim getCudaBlockDim(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SketchBlockDim")) {
            return getCudaBlockDimFromSketchBlockDim(node);
        } else {
            throw new RuntimeException("no way to return a 'CudaBlockDim' from a node of type " + typ);
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

    public ExprNew getExprNew(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKNew")) {
            return getExprNewFromSKNew(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprNew' from a node of type " + typ);
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

    public StmtVarDecl getStmtVarDecl(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ValDef")) {
            return getStmtVarDeclFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtVarDecl' from a node of type " + typ);
        }
    }

    public StmtWhile getStmtWhile(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKWhileLoop")) {
            return getStmtWhileFromSKWhileLoop(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtWhile' from a node of type " + typ);
        }
    }

    public ExprField getExprField(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("FieldAccess")) {
            return getExprFieldFromFieldAccess(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprField' from a node of type " + typ);
        }
    }

    public StmtExpr getStmtExpr(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SKStmtExpr")) {
            return getStmtExprFromSKStmtExpr(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtExpr' from a node of type " + typ);
        }
    }

    public ExprConstant getExprConstant(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else if (typ.equals("BooleanConstant")) {
            return getExprConstBooleanFromBooleanConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprConstant' from a node of type " + typ);
        }
    }

    public ExprTprint getExprTprint(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SketchTprintCall")) {
            return getExprTprintFromSketchTprintCall(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprTprint' from a node of type " + typ);
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

    public CudaSyncthreads getCudaSyncthreads(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SyncthreadsCall")) {
            return getCudaSyncthreadsFromSyncthreadsCall(node);
        } else {
            throw new RuntimeException("no way to return a 'CudaSyncthreads' from a node of type " + typ);
        }
    }

    public StmtEmpty getStmtEmpty(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getStmtEmptyFromUnitConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'StmtEmpty' from a node of type " + typ);
        }
    }

    public TprintTuple getTprintTuple(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SketchPrintTuple")) {
            return getTprintTupleFromSketchPrintTuple(node);
        } else {
            throw new RuntimeException("no way to return a 'TprintTuple' from a node of type " + typ);
        }
    }

    public CudaThreadIdx getCudaThreadIdx(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SketchThreadIdx")) {
            return getCudaThreadIdxFromSketchThreadIdx(node);
        } else {
            throw new RuntimeException("no way to return a 'CudaThreadIdx' from a node of type " + typ);
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

    public Expression getExpression(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else if (typ.equals("BooleanConstant")) {
            return getExprConstBooleanFromBooleanConstant(node);
        } else if (typ.equals("SketchBlockDim")) {
            return getCudaBlockDimFromSketchBlockDim(node);
        } else if (typ.equals("SketchThreadIdx")) {
            return getCudaThreadIdxFromSketchThreadIdx(node);
        } else if (typ.equals("SketchTprintCall")) {
            return getExprTprintFromSketchTprintCall(node);
        } else if (typ.equals("SketchArrayAccess")) {
            return getExprArrayRangeFromSketchArrayAccess(node);
        } else if (typ.equals("NewArray")) {
            return getExprArrayInitFromNewArray(node);
        } else if (typ.equals("SKNew")) {
            return getExprNewFromSKNew(node);
        } else if (typ.equals("NullTypeConstant")) {
            return getExprNullPtrFromNullTypeConstant(node);
        } else if (typ.equals("FieldAccess")) {
            return getExprFieldFromFieldAccess(node);
        } else if (typ.equals("FcnCall")) {
            return getExprFunCallFromFcnCall(node);
        } else if (typ.equals("FcnCallUnaryNegative")) {
            return getExprUnaryFromFcnCallUnaryNegative(node);
        } else if (typ.equals("VarRef")) {
            return getExprVarFromVarRef(node);
        } else if (typ.equals("UnitConstant")) {
            return getExprConstUnitFromUnitConstant(node);
        } else if (typ.equals("IntConstant")) {
            return getExprConstIntFromIntConstant(node);
        } else if (typ.equals("BooleanConstant")) {
            return getExprConstBooleanFromBooleanConstant(node);
        } else if (typ.equals("HoleCall")) {
            return getExprStarFromHoleCall(node);
        } else if (typ.equals("FcnBinaryCall")) {
            return getExprBinaryFromFcnBinaryCall(node);
        } else {
            throw new RuntimeException("no way to return a 'Expression' from a node of type " + typ);
        }
    }

    public ExprConstBoolean getExprConstBoolean(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("BooleanConstant")) {
            return getExprConstBooleanFromBooleanConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprConstBoolean' from a node of type " + typ);
        }
    }

    public String getString(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("PrintName")) {
            return getStringFromPrintName(node);
        } else {
            throw new RuntimeException("no way to return a 'String' from a node of type " + typ);
        }
    }

    public FieldDecl getFieldDecl(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("ValDef")) {
            return getFieldDeclFromValDef(node);
        } else {
            throw new RuntimeException("no way to return a 'FieldDecl' from a node of type " + typ);
        }
    }

    public ExprNullPtr getExprNullPtr(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("NullTypeConstant")) {
            return getExprNullPtrFromNullTypeConstant(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprNullPtr' from a node of type " + typ);
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

    public ExprArrayRange getExprArrayRange(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("SketchArrayAccess")) {
            return getExprArrayRangeFromSketchArrayAccess(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprArrayRange' from a node of type " + typ);
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

    public CudaMemoryType getCudaMemoryType(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("CudaMemLocal")) {
            return getCudaMemoryTypeFromCudaMemLocal(node);
        } else if (typ.equals("CudaMemGlobal")) {
            return getCudaMemoryTypeFromCudaMemGlobal(node);
        } else if (typ.equals("CudaMemDefaultShared")) {
            return getCudaMemoryTypeFromCudaMemDefaultShared(node);
        } else if (typ.equals("CudaMemImplicitShared")) {
            return getCudaMemoryTypeFromCudaMemImplicitShared(node);
        } else if (typ.equals("CudaMemShared")) {
            return getCudaMemoryTypeFromCudaMemShared(node);
        } else {
            throw new RuntimeException("no way to return a 'CudaMemoryType' from a node of type " + typ);
        }
    }

    public ExprFunCall getExprFunCall(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("FcnCall")) {
            return getExprFunCallFromFcnCall(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprFunCall' from a node of type " + typ);
        }
    }

    public ExprArrayInit getExprArrayInit(final GXLNode node) {
        String typ = GxlImport.nodeType(node);
        if (typ.equals("NewArray")) {
            return getExprArrayInitFromNewArray(node);
        } else {
            throw new RuntimeException("no way to return a 'ExprArrayInit' from a node of type " + typ);
        }
    }
}
