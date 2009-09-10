package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.compiler.ast.{base, core, scala => scast}

import scala.tools.nsc
import nsc._

case class MessageProxy(val message : String) extends base.FEAnyNode

/**
 * Main map from Scala AST nodes to SKETCH AST nodes.
 * Bugs are most likely in the SketchTypes or SketchNames files
 * or in SKETCH lowering. This code is rather straightforward, since
 * I add a bunch of proxy nodes to the SKETCH AST.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class ScalaSketchNodeMap {
    val global : Global
    val types : SketchTypes
    val ctx : core.FENode

    import global._
    import types._

    val goto_connect : AutoNodeConnector[Symbol]
    val class_connect : AutoNodeConnector[Symbol]
    val class_fcn_connect : AutoNodeConnector[Symbol]

    def subtree(tree : Tree, next_info : ContextInfo = null) : base.FEAnyNode
    def subarr(arr : List[Tree]) : Array[base.FEAnyNode]
    def gettype(tpe : Type) : core.typs.Type
    def gettype(tree : Tree) : core.typs.Type
    def getname(elt : Object, sym : Symbol) : String
    def getname(elt : Object) : String

    def unaryExpr(code : Int, target : core.exprs.Expression) : core.exprs.ExprUnary = {
        val sc = scalaPrimitives
        import core.exprs.{ExprUnary => sk}

        code match {
            case sc.ZNOT => new core.exprs.ExprUnary(ctx, sk.UNOP_NOT, target)
            case sc.NOT => new core.exprs.ExprUnary(ctx, sk.UNOP_BNOT, target)
            case sc.NEG => new core.exprs.ExprUnary(ctx, sk.UNOP_NEG, target)
        }
    }

    def binaryExpr(code : Int, left : core.exprs.Expression, right : core.exprs.Expression)
        : core.exprs.ExprBinary =
    {
        val sc = scalaPrimitives
        import core.exprs.{ExprBinary => sk}

        code match {
            case sc.ADD => new core.exprs.ExprBinary(ctx, sk.BINOP_ADD, left, right)
            case sc.SUB => new core.exprs.ExprBinary(ctx, sk.BINOP_SUB, left, right)
            case sc.MUL => new core.exprs.ExprBinary(ctx, sk.BINOP_MUL, left, right)
            case sc.DIV => new core.exprs.ExprBinary(ctx, sk.BINOP_DIV, left, right)
            case sc.MOD => new core.exprs.ExprBinary(ctx, sk.BINOP_MOD, left, right)

            case sc.EQ | sc.ID => new core.exprs.ExprBinary(ctx, sk.BINOP_EQ, left, right)
            case sc.NE | sc.NI => new core.exprs.ExprBinary(ctx, sk.BINOP_NEQ, left, right)
            case sc.LT => new core.exprs.ExprBinary(ctx, sk.BINOP_LT, left, right)
            case sc.LE => new core.exprs.ExprBinary(ctx, sk.BINOP_LE, left, right)
            case sc.GT => new core.exprs.ExprBinary(ctx, sk.BINOP_GT, left, right)
            case sc.GE => new core.exprs.ExprBinary(ctx, sk.BINOP_GE, left, right)

            case sc.AND => new core.exprs.ExprBinary(ctx, sk.BINOP_BAND, left, right)
            case sc.OR => new core.exprs.ExprBinary(ctx, sk.BINOP_BOR, left, right)
            case sc.XOR => new core.exprs.ExprBinary(ctx, sk.BINOP_BXOR, left, right)

            case sc.LSL => new core.exprs.ExprBinary(ctx, sk.BINOP_LSHIFT, left, right)
            case sc.ASR => new core.exprs.ExprBinary(ctx, sk.BINOP_RSHIFT, left, right)

            case _ => assertFalse("bad arithmetic op", code.toString, left, right)
        }
    }

    def arrayExpr(code : Int, target : core.exprs.Expression,
            args : Array[base.FEAnyNode]) : base.FEAnyNode =
    {
        val sc = scalaPrimitives

        if (sc.isArrayNew(code)) {
            not_implemented("array new")
        } else if (sc.isArrayLength(code)) {
            not_implemented("array length")
        } else if (sc.isArrayGet(code)) (args match {
            case Array(idx) => new core.exprs.ExprArrayRange(target, idx)
            case _ => not_implemented("array get with unexpected args", args)
        }) else if (sc.isArraySet(code)) (args match {
            case Array(idx, expr) => new core.stmts.StmtAssign(ctx,
                new core.exprs.ExprArrayRange(target, idx), expr)
            case _ => not_implemented("array set with unexpected args", args)
        }) else {
            assertFalse("no matching array opcode for code", code : Integer)
        }
    }

    /**
     * NOTE - main translation method.
     */
    def execute(tree : Tree, info : ContextInfo) : base.FEAnyNode = {
        val treesym = tree.symbol
        tree match {
            // much code directly copied from GenICode.scala (BSD license);
            // that file is a lot more complete though.
            case Apply(fcn @ Select(Super(_, mix), _), args) =>
                val fcnsym = fcn.symbol
                DebugOut.print(">>> super symbol", treesym.toString)
                DebugOut.print(">>> is module symbol", treesym.isModuleClass.toString)
                val ths_var = class_connect.connect_from(treesym,
                    new scast.exprs.vars.ScalaThis(ctx))
                class_fcn_connect.connect_from(fcnsym,
                        new scast.exprs.ScalaClassFunctionCall(ctx, ths_var, subarr(args)) )

            case Apply(fcn @ Select(New(tpt), nme.CONSTRUCTOR), args) =>
                val ctor_sym = fcn.symbol

                import global.icodes._
                toTypeKind(tpt.tpe) match {
                    case arr : ARRAY => not_implemented("array constructor")
                    case ref_typ @ REFERENCE(cls_sym) =>
                        assert(ctor_sym.owner == cls_sym, "new sym owner not class")
                        val thisobj = class_connect.connect_from(cls_sym,
                            new scast.exprs.ScalaExprClsNew(ctx))
                        thisobj.init_call = class_fcn_connect.connect_from(ctor_sym,
                            new scast.exprs.ScalaClassFunctionCall(ctx, thisobj, subarr(args)))
                        thisobj
                    case _ => not_implemented("unknown new usage")
                }

            case Apply(fcn, args) =>
                val fcnsym = fcn.symbol
                if (fcnsym.isLabel) {
                    return goto_connect.connect_from(fcnsym, new scast.misc.ScalaGotoCall(ctx))
                }

                // otherwise, it's of the form (object or class).(function name)
                val Select(target_tree, fcn_name) = fcn
                val target = subtree(target_tree)
                if (scalaPrimitives.isPrimitive(fcnsym)) {
                    val code = scalaPrimitives.getPrimitive(fcnsym, target_tree.tpe)
                    if (scalaPrimitives.isArrayOp(code)) {
                        arrayExpr(code, target, subarr(args))
                    } else (args match {
                        case Nil => unaryExpr(code, target)
                        case right :: Nil => binaryExpr(code, target, subtree(right))
                        case _ => assertFalse("bad argument list for arithmetic op", target,
                            code : java.lang.Integer, args)
                    })
                } else if (fcnsym.isStaticMember) {
                    new core.exprs.ExprFunCall(ctx, getname(fcnsym.name), subarr(args))
                } else if (fcnsym.isClassConstructor) {
                    not_implemented("class constructor call", fcnsym.toString, args.toString)
                } else {
                    class_fcn_connect.connect_from(fcnsym,
                        new scast.exprs.ScalaClassFunctionCall(ctx, target, subarr(args)))
                }

            case ArrayValue(elemtpt, elems) =>
                new core.exprs.ExprArrayInit(ctx, subarr(elems))

            case Assign(lhs, rhs) =>
                new core.stmts.StmtAssign(ctx, subtree(lhs), subtree(rhs))

            case Bind(name, body) =>
                new scast.stmts.ScalaBindStmt(
                    ctx, gettype(body), getname(name), subtree(body))

            case Block(stmt_arr, expr) =>
                new scast.stmts.ScalaBlock(ctx, subarr(stmt_arr), subtree(expr))

            case CaseDef(pat, guard, body) =>
                new scast.stmts.ScalaCaseStmt(
                    ctx, subtree(pat), subtree(guard), subtree(body))

            case ClassDef(mods, name, tparams, impl) =>
                val next_info = new ContextInfo(info)
                next_info.curr_clazz = new scast.typs.ScalaClass(
                    ctx, getname(name), subarr(tparams))
                next_info.clazz_symbol = treesym
                class_connect.connect_to(treesym, next_info.curr_clazz)
                subtree(impl, next_info)
                next_info.curr_clazz

            case DefDef(mods, name, tparams, vparamss, tpe, body) =>
                // info.curr_clazz
                val params = vparamss match {
                    case Nil => List[ValDef]()
                    case vparams :: Nil => vparams
                    case _ =>
                        assertFalse("unknown defdef params", vparamss)
                }
                // add the return node
                val body_stmt = (subtree(body) match {
                    case stmt : core.stmts.Statement => stmt
                    case expr : core.exprs.Expression => new core.stmts.StmtReturn(ctx, expr)
                })
                class_fcn_connect.connect_to(treesym,
                    new scast.misc.ScalaClassFunction(ctx, core.Function.FUNC_STATIC,
                        treesym.isStaticMember,
                        info.curr_clazz,
                        getname(name), gettype(tpe), subarr(params), body_stmt))

            case Ident(name) =>
                new core.exprs.ExprVar(ctx, getname(name))

            case If(cond, thenstmt, elsestmt) => new core.stmts.StmtIfThen(
                    ctx, subtree(cond), subtree(thenstmt), subtree(elsestmt))

            case LabelDef(name, params, rhs) =>
                goto_connect.connect_to(treesym, new scast.misc.ScalaGotoLabel(
                    ctx, getname(name), subarr(params), subtree(rhs)))

            case Literal(Constant(value)) => value match {
                    case v : Boolean => new core.exprs.ExprConstBoolean(ctx, v)
                    case v : Char => new core.exprs.ExprConstChar(ctx, v)
                    case v : Float => new core.exprs.ExprConstFloat(ctx, v)
                    case v : Int => new core.exprs.ExprConstInt(ctx, v)
                    case v : String => new core.exprs.ExprConstStr(ctx, v)
                    case () => new scast.exprs.ScalaUnitExpression(ctx)
                    case _ => if (value == null) {
                        new core.exprs.ExprNullPtr(ctx)
                    } else {
                        not_implemented("scala constant literal", value.toString)
                    }
                }

            case Match(selector, cases) =>
                new scast.stmts.ScalaMatchStmt(
                    ctx, subtree(selector), subarr(cases))

            case New(tpt : Tree) =>
                new core.exprs.ExprNew(ctx, gettype(tpt))

            case PackageDef(pid, stats) =>
                DebugOut.print("stats", subarr(stats))
                new scast.misc.ScalaPackageDef()

            case Return(expr) =>
                new core.stmts.StmtReturn(ctx, subtree(expr))

            case Select(qualifier, name) =>
                if (treesym.isModule) {
                    gettype(treesym.tpe)
                } else {
                    new core.exprs.ExprField(ctx, subtree(qualifier), getname(name))
                }

            case Super(qual, mix) =>
                new scast.exprs.vars.ScalaSuperRef(ctx, gettype(tree))

            case Template(parents, self, body) =>
                DebugOut.print("not visiting parents", parents)
                for (sketch_node <- subarr(body)) sketch_node match {
                    case f : scast.misc.ScalaClassFunction => ()
                    case variable : core.stmts.StmtVarDecl =>
                        assert(variable.getNumVars == 1,
                            "multi-variable declarations not allowed")
                        assert(variable.getInit(0).isInstanceOf[scast.exprs.ScalaEmptyExpression],
                            "all variables in the class body should initially " +
                            "be assigned to the empty expression. class given: " +
                            variable.getInit(0).getClass)
                        info.curr_clazz.variables.add(variable)
                    case _ =>
                        not_implemented("element", "'" + sketch_node + "'",
                            "in class body")
                }
                null

            // qual may reference an outer class.
            case This(qual) =>
                DebugOut.print(">>> this symbol", treesym.toString)
                DebugOut.print(">>> is module symbol", treesym.isModuleClass.toString)
                DebugOut.print(">>> is package class", treesym.isPackageClass.toString)
                if (treesym.isModuleClass && treesym != info.clazz_symbol) {
                    if (treesym.isPackageClass) {
                        not_implemented("package class")
//                         DebugOut.print("WARNING - package class, returning null")
                    } else {
                        not_implemented("other symbol...")
                    }
                    null
                } else {
                    class_connect.connect_from(treesym,
                        new scast.exprs.vars.ScalaThis(ctx))
                }

            case Throw(expr) =>
                new scast.stmts.ScalaThrow(ctx, subtree(expr))

            case Try(block, catches, finalizer) =>
                new scast.stmts.ScalaTryCatchFinally(
                    ctx, subtree(block), subarr(catches), subtree(finalizer))

            case TypeApply(fcn, args) =>
                DebugOut.not_implemented("type apply", subtree(fcn), subarr(args))
                new scast.exprs.ScalaTypeApply(ctx, null, null)

            case TypeTree() => gettype(tree)

            case Typed(expr, typ) =>
                new scast.exprs.ScalaTypedExpression(
                    ctx, subtree(expr), gettype(typ))

            case ValDef(mods, name, typ, rhs) => new core.stmts.StmtVarDecl(
                ctx, gettype(typ), getname(name), subtree(rhs))

            case EmptyTree =>
                new scast.exprs.ScalaEmptyExpression(ctx)

            case _ =>
                not_implemented("didn't match Scala node", tree)
        }
    }
}
