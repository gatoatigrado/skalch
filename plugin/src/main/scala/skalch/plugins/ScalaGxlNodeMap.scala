package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._

import scala.tools.nsc
import nsc._
import nsc.util.{FakePos, OffsetPosition, RangePosition}

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
abstract class ScalaGxlNodeMap() extends NodeFactory {
    import _glbl._
    var visited : HashSet[Tree] = null

    // === String representation of node types ===
    // All names should be unambiguous, and immediately identifiable

    def unaryExpr(code : Int) : String = "Unary" + (code match {
        case scalaPrimitives.ZNOT => "Nonzero"
        case scalaPrimitives.NOT => "InvertBits"
        case scalaPrimitives.POS => "Positive"
        case scalaPrimitives.NEG => "Negative"
        case scalaPrimitives.C2I => "IntCast"
        case _ => not_implemented("other unary expr", code.toString)
    })

    def binaryExpr(code : Int) : String = {
        import scalaPrimitives._

        val map = Map(
            ADD -> "Add", SUB -> "Subtract", MUL -> "Multiply",
            DIV -> "Divide", MOD -> "Modulo",

            // object equality versus reference identity
            EQ -> "PrimitiveEquals", NE -> "PrimitiveNotEquals",
            ID -> "SameObj", NI -> "NotSameObj",

            // ordered set operations
            LT -> "LessThan", LE -> "LessThanOrEqual",
            GT -> "GreaterThan", GE -> "GreaterThanOrEqual",

            // logical / bitwise ops
            AND -> "BitwiseAnd", OR -> "BitwiseOr", XOR -> "BitwiseXor",
            LSL -> "BitwiseShiftLeft",
            LSR -> "ArithmeticShiftLeft",
            ASR -> "ArithmeticShiftRight",

            // logic operations
            ZOR -> "LogicOr", ZAND -> "LogicAnd",

            // others which aren't supported in SKETCH
            CONCAT -> "StringConcat"
            )
        "Binary" + map(code)
    }

    def arrayExpr(code : Int) : String = {
        if (scalaPrimitives.isArrayNew(code)) {
            "ArrayNew"
        } else if (scalaPrimitives.isArrayLength(code)) {
            "ArrayLength"
        } else if (scalaPrimitives.isArrayGet(code)) {
            "ArrayGet"
        } else if (scalaPrimitives.isArraySet(code)) {
            "ArraySet"
        } else {
            not_implemented("arrayExpr(): couldn't find anything for code " + code)
        }
    }




    // === boilerplate ===

    def pos_to_tuple(pos : Position) = pos match {
        case NoPosition => new SimplePosition(0, 0)
        case FakePos(msg) => new SimplePosition(0, 0)
        case t : Position => new SimplePosition(t.line, t.column)
    }

    /**
     * NOTE - main translation method.
     */
    def getGxlAST(tree : Tree) : GrNode = {
//         println("=== gxlast ===")
//         println("    " + tree + ", " + tree.getClass)
//         println()
        visited.add(tree)

        val clsname = tree.getClass().getSimpleName() match {
            case "Apply" => "FcnCall"
            case "Ident" => "VarRef"
            case "Select" =>
                if (tree.symbol.isModule) "ClassRef" else "FieldAccess"
            case "DefDef" => "FcnDef"
            case "ArrayValue" => "NewArray"
            case "Trees$EmptyTree$" => "EmptyTree"
            case other => other
        }
        val start = pos_to_tuple(tree.pos.focusStart)
        var node = new GrNode(clsname, "line_" + start.line + "_" + id_ctr(),
            start, pos_to_tuple(tree.pos.focusEnd))
        node.append_str_attr("scalaSource", tree.toString())

        /** closure functions */
        val node_fcns = new BoundNodeFcns(node, clsname)
        import node_fcns._

        symlink(clsname, tree.symbol)
        symlink(clsname + "Type", tree.tpe.typeSymbol)
        symlink(clsname + "Term", tree.tpe.termSymbol)
        tree.tpe.normalize match {
            case TypeRef(pre, sym, List(inner_typ)) =>
                symlink("InnerType", inner_typ.typeSymbol); ()
            case _ => ()
        }

        /** the actual match statement */
        tree match {
            // === package / class definitions ===

            case PackageDef(pid, stats) =>
                subarr("PackageDefElement", stats)

            case ClassDef(mods, name, tparams, impl) =>
                subchain("ClassDefTypeParams", tparams)
                subtree("ClassDefImpl", impl)
                for (fldsym <- tree.symbol.info.decls.iterator)
                    if (!fldsym.isMethod && fldsym.isTerm)
                        symlink("ClassDefField", fldsym)

            case Template(parents, self, body) =>
                subarr("TemplateElement", body)



            // === function and jump related ===

            case DefDef(mods, name, tparams, vparamss, tpe, body) =>
                val params = vparamss match {
                    case Nil => List[ValDef]()
                    case vparams :: Nil => vparams
                    case _ =>
                        assertFalse("unknown defdef params", vparamss)
                }
                // add the return node
                subtree("FcnBody", body)
                subchain("FcnDefParams", params)
                symlink(clsname + "ReturnType", tree.symbol.info.resultType.typeSymbol)

            case Apply(fcn @ Select(Super(_, mix), _), args) =>
                assert(!tree.symbol.isModuleClass, "not implemented")
//                 println(">>> super symbol", tree.symbol.toString)
//                 println(">>> is module symbol", tree.symbol.isModuleClass.toString)
                visited.add(fcn)
                node.set_type("FcnSuperCall", "FcnCall")
                subchain("FcnArg", args)

            case Apply(fcn @ Select(New(tpt), nme.CONSTRUCTOR), args) =>
                visited.add(fcn)
                subchain("FcnArg", args)

                import _glbl.icodes._
                toTypeKind(tpt.tpe) match {
                    case arr : ARRAY => node.set_type("NewArrayCall", "FcnCall")
                    case ref_typ @ REFERENCE(cls_sym) =>
                        assert(fcn.symbol.owner == cls_sym, "new sym owner not class")
                        node.set_type("NewConstructor", "FcnCall")
                        symlink("NewClass", cls_sym)
                    case _ => not_implemented("unknown new usage")
                }

            case Apply(fcn, args) =>
                visited.add(fcn)
                subchain("FcnArg", args)

                val fcnsym = fcn.symbol
                if (fcn.symbol.isLabel) {
                    node.set_type("GotoCall", null)
                } else {
                    // otherwise, it's of the form (object or class).(function name)
                    fcn match {
                        case Select(target_tree, fcn_name) =>
                            subtree("FcnTarget", target_tree)
                            if (scalaPrimitives.isPrimitive(fcn.symbol)) {
                                val code = scalaPrimitives.getPrimitive(fcn.symbol, target_tree.tpe)
                                node.set_type("FcnCall" + (if (scalaPrimitives.isArrayOp(code)) {
                                    arrayExpr(code)
                                } else (args match {
                                    case Nil => unaryExpr(code)
                                    case right :: Nil => binaryExpr(code)
                                    case _ => assertFalse("bad argument list for arithmetic op",
                                        target_tree, code : java.lang.Integer, args)
                                })), "FcnCall")
                            } else if (fcn.symbol.isStaticMember) {
                                node.set_type("StaticFcnCall", "FcnCall")
                            } else if (fcn.symbol.isClassConstructor) {
                                node.set_type("ClassConstructorCall", "FcnCall")
                            }
                        case TypeApply(fcn, type_args) =>
                            subtree("FcnTarget", fcn)
                            subchain("FcnCallTypeArgs", type_args)
                            node.set_type("FcnCallTypeApply", "FcnCall")
                        case other =>
                            not_implemented("fcn call " + other, "type", fcn.getClass)
                            not_implemented("fcn call type " + other)
                    }
                }

            case LabelDef(name, params, rhs) =>
                subtree("LabelDefRhs", rhs)

            case Try(block, catches, finalizer) =>
                subtree("Try", block)
                subchain("Catch", catches)
                subtree("Finally", finalizer)



            // === expressions ===

            case ArrayValue(elemtpt, elems) =>
                subchain("ArrValue", elems)

            case Ident(name) => ()
//                 symlink("Var", tree.symbol)

            case Select(qualifier, name) =>
                if (tree.symbol.isModule) {
                    // FIXME -- not quite...
                    node.set_type("QualifiedClassRef", null)
//                     symlink("Class", tree.symbol)
//                     gettype(tree.symbol.tpe)
                } else {
                    assert (node.typ == "FieldAccess",
                            "(ScalaGxlNodeMap.scala) FieldAccess not named correctly")
                    subtree("FieldAccessObject", qualifier)
//                     new core.exprs.ExprField(ctx, , getname(name))
                }

            case This(qual) =>
                if (tree.symbol.isPackageClass) {
                    not_implemented("package class")
                }

            case Throw(expr) =>
                subtree("ThrowExpr", expr)

            case Literal(Constant(value)) =>
                value match {
                    case v : Boolean =>
                        node.set_type("BooleanConstant", "Constant")
                        node.append_attr("value", new GXLBool(v))
                    case v : Char =>
                        node.set_type("CharConstant", "Constant")
                        node.append_attr("value", new GXLString(v.toString))
                    case v : Float =>
                        node.set_type("FloatConstant", "Constant")
                        node.append_attr("value", new GXLFloat(v))
                    case v : Short =>
                        node.set_type("ShortConstant", "Constant")
                        node.append_attr("value", new GXLInt(v))
                    case v : Int =>
                        node.set_type("IntConstant", "Constant")
                        node.append_attr("value", new GXLInt(v))
                    case v : Long =>
                        node.set_type("LongConstant", "Constant")
                        node.append_attr("value", new GXLInt(v.asInstanceOf[Int]))
                    case v : String =>
                        node.set_type("StringConstant", "Constant")
                        node.append_attr("value", new GXLString(v))
                    case () =>
                        node.set_type("UnitConstant", "Constant")
                    case null =>
                        node.set_type("NullTypeConstant", "Constant")
                    case r : AnyRef =>
                        not_implemented("scala constant literal", r.getClass().getName(),  value.toString)
                    case _ =>
                        not_implemented("scala constant literal", value.toString)
                }

            case EmptyTree => ()

            case Bind(name, body) =>
                subtree("BindBody", body)

            // === branching ===

            case If(cond, thenstmt, elsestmt) =>
                subtree("IfCond", cond)
                subtree("IfThen", thenstmt)
                subtree("IfElse", elsestmt)

            case Match(selector, cases) =>
                subtree("MatchTarget", selector)
                subchain("MatchCase", cases)

            case CaseDef(pat, guard, body) =>
                subtree("CasePattern", pat)
                subtree("CaseGuard", guard)
                subtree("CaseBody", body)

            case Typed(expr, typ) =>
                subtree("TypedExpression", expr)

            case TypeTree() => ()



            // === statements ===

            case Assign(lhs, rhs) =>
                subtree("AssignLhs", lhs)
                subtree("AssignRhs", rhs)

            case Block(stmt_arr, expr) =>
                subchain("BlockStmt", stmt_arr)
                subtree("BlockExpr", expr)

            case ValDef(mods, name, typ, rhs) =>
                subtree("ValDefRhs", rhs)

            case Return(expr) =>
                subtree("ReturnExpr", expr)

            case _ =>
                not_implemented("ERROR -- Scala construct not supported:", tree.getClass)
        }
        node.accept_type()
        node
    }
}
