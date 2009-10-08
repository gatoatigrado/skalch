package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.compiler.ast.{base, core, scala => scast}

import scala.tools.nsc
import nsc._
import nsc.util.{Position, NoPosition, FakePos, OffsetPosition, RangePosition}

import net.sourceforge.gxl._

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
abstract class ScalaGxlNodeMap() {
    val _glbl : Global

    import _glbl._

    object nf extends {
        val global : _glbl.type = _glbl
    } with NodeFactory
    import nf._

    /*
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
            assertFalse("no matching array opcode for code", code : java.lang.Integer)
        }
    }
    */

    def pos_to_tuple(pos : Position) = pos match {
        case NoPosition => (0, 0)
        case FakePos(msg) => (0, 0)
        case t : Position => (t.line, t.column)
    }

    /**
     * NOTE - main translation method.
     */
    def getGxlAST(tree : Tree) : GrNode = {
        start = pos_to_tuple(tree.pos.focusStart)
        end = pos_to_tuple(tree.pos.focusEnd)

        val clsname = tree.getClass().getName() match {
            case "Apply" => "FcnCall"
            case "Ident" => "Var"
            case "ClassDef" => "Class"
            case "DefDef" => "FcnDef"
            case other => other
        }
        var node = GrNode(clsname, "line_" + start._1 + "_" + id_ctr())

        /** closure functions */
        def subtree(edge_typ : String, subtree : Tree) =
            GrEdge(node, edge_typ, getGxlAST(subtree))
        def subarr(edge_typ : String, arr : List[Tree]) {
            for (v <- arr) {
                GrEdge(node, edge_typ, getGxlAST(v))
            }
        }
        def subchain(edge_typ : String, arr : List[Tree]) = arr match {
            case hd :: tail_ =>
                GrEdge(node, nme + "symbol", getGxlAST(hd))
                var last_node = node
                for (v <- (tail_ map getGxlAST)) {
                    GrEdge(last_node, edge_typ + "Next", v)
                    last_node = v
                }
            case Nil => ()
            case _ => ()
        }
        def symlink(nme : String, sym : Symbol) =
            GrEdge(node, nme + "Symbol", getsym(sym))

        if (tree.symbol != null) {
            symlink(clsname, tree.symbol)
        }

        tree match {
            // === package / class definitions ===

            case PackageDef(pid, stats) =>
                subarr("PackageDefElement", stats)

            case ClassDef(mods, name, tparams, impl) =>
                subchain("TypeParams", tparams)
                subtree("Impl", impl)

            case Template(parents, self, body) =>
                subarr("Element", body)



            // === function and jump related ===

            case DefDef(mods, name, tparams, vparamss, tpe, body) =>
                // info.curr_clazz
                val params = vparamss match {
                    case Nil => List[ValDef]()
                    case vparams :: Nil => vparams
                    case _ =>
                        assertFalse("unknown defdef params", vparamss)
                }
                // add the return node
                subtree("Body", body)

            case Apply(fcn @ Select(Super(_, mix), _), args) =>
                DebugOut.print(">>> super symbol", tree.symbol.toString)
                DebugOut.print(">>> is module symbol", tree.symbol.isModuleClass.toString)
                node.typ = "FcnSuperCall"
                subarr("FcnArgs", args)
                not_implemented("check array args", mix.toString, mix.getClass)

            case Apply(fcn @ Select(New(tpt), nme.CONSTRUCTOR), args) =>
                symlink("FcnCtorCall", fcn.symbol)
                subarr("FcnArgs", args)

                import global.icodes._
                toTypeKind(tpt.tpe) match {
                    case arr : ARRAY => node.typ = "NewArrayCall"
                    case ref_typ @ REFERENCE(cls_sym) =>
                        assert(fcn.symbol.owner == cls_sym, "new sym owner not class")
                        node.typ = "NewConstructor"
                        symlink("NewClass", cls_sym)
                    case _ => not_implemented("unknown new usage")
                }

            case Apply(fcn, args) =>
                symlink("Fcn", fcn.symbol)
                subarr("FcnArgs", args)

                val fcnsym = fcn.symbol
                if (fcn.symbol.isLabel) {
                    not_implemented("label")
//                     return goto_connect.connect_from(fcnsym, new
//                         scast.misc.ScalaGotoCall(ctx)
                } else {
                    // otherwise, it's of the form (object or class).(function name)
                    val Select(target_tree, fcn_name) = fcn
                    val target = getGxlAST(target_tree)
                    if (scalaPrimitives.isPrimitive(fcn.symbol)) {
                        val code = scalaPrimitives.getPrimitive(fcn.symbol, target_tree.tpe)
                        not_implemented(code.toString)
//                         if (scalaPrimitives.isArrayOp(code)) {
//                             arrayExpr(code, target, subarr(args))
//                         } else (args match {
//                             case Nil => unaryExpr(code, target)
//                             case right :: Nil => binaryExpr(code, target, subtree(right))
//                             case _ => assertFalse("bad argument list for arithmetic op", target,
//                                 code : java.lang.Integer, args)
//                         })
                    } else if (fcn.symbol.isStaticMember) {
                        node.typ = "StaticFcnCall"
                    } else if (fcn.symbol.isClassConstructor) {
                        node.typ = "ClassConstructorCall"
                    } else {
                        node.typ = "FcnCall"
                    }
                }



            // === expressions ===

            case Ident(name) =>
                symlink("Var", tree.symbol)

            case Select(qualifier, name) =>
                if (tree.symbol.isModule) {
                    // FIXME -- not quite...
                    node.typ = "QualifiedClass"
                    symlink("Class", tree.symbol)
//                     gettype(tree.symbol.tpe)
                } else {
                    node.typ = "FieldAccess"
                    subtree("FieldAccessObject", qualifier)
//                     new core.exprs.ExprField(ctx, , getname(name))
                }

            case This(qual) =>
                if (tree.symbol.isPackageClass) {
                    not_implemented("package class")
                }

            case Literal(Constant(value)) =>
                node.attrs.append( ("value", value) )
                value match {
                    case v : Boolean => node.typ = "BooleanConstant"
                    case v : Char => node.typ = "CharConstant"
                    case v : Float => node.typ = "FloatConstant"
                    case v : Int => node.typ = "IntConstant"
                    case v : String => node.typ = "StringConstant"
                    case () => new scast.exprs.ScalaUnitExpression(ctx)
                    case _ => if (value == null) {
                        new core.exprs.ExprNullPtr(ctx)
                    } else {
                        not_implemented("scala constant literal", value.toString)
                    }
                }



            // === statements ===

            case Block(stmt_arr, expr) =>
                subchain("BlockStmt", stmt_arr)
                subtree("BlockExpr", expr)

            case ValDef(mods, name, typ, rhs) =>
                subtree("VarDeclRhs", rhs)

            case _ =>
                not_implemented("didn't match Scala node", tree.getClass)
        }
        node
    }
}
