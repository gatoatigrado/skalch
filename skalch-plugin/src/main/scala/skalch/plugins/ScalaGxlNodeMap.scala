package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
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

    var visited : HashSet[Tree] = null
    var gxldoc : GXLDocument = null

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
        return "Binary" + map(code)
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
        case NoPosition => (0, 0)
        case FakePos(msg) => (0, 0)
        case t : Position => (t.line, t.column)
    }

    /**
     * NOTE - main translation method.
     */
    def getGxlAST(tree : Tree) : GrNode = {
//         println("=== gxlast ===")
//         println("    " + tree + ", " + tree.getClass)
//         println()
        visited.add(tree)
        start = pos_to_tuple(tree.pos.focusStart)
        end = pos_to_tuple(tree.pos.focusEnd)

        val clsname = tree.getClass().getName() match {
            case "Apply" => "FcnCall"
            case "Ident" => "Var"
            case "Select" => "ClassRef"
            case "ClassDef" => "Class"
            case "DefDef" => "FcnDef"
            case "ArrayValue" => "NewArray"
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

        /** the actual match statement */
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
                subchain("Params", params)

            case Apply(fcn @ Select(Super(_, mix), _), args) =>
                assert(!tree.symbol.isModuleClass, "not implemented")
//                 println(">>> super symbol", tree.symbol.toString)
//                 println(">>> is module symbol", tree.symbol.isModuleClass.toString)
                visited.add(fcn)
                node.typ = "FcnSuperCall"
                symlink("FcnSuperCall", fcn.symbol)
                subarr("FcnArgs", args)

            case Apply(fcn @ Select(New(tpt), nme.CONSTRUCTOR), args) =>
                visited.add(fcn)
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
                visited.add(fcn)
                symlink("Fcn", fcn.symbol)
                subarr("FcnArgs", args)

                val fcnsym = fcn.symbol
                if (fcn.symbol.isLabel) {
                    node.typ = "GotoCall"
                } else {
                    // otherwise, it's of the form (object or class).(function name)
                    fcn match {
                        case Select(target_tree, fcn_name) =>
                            subtree("FcnTarget", target_tree)
                            if (scalaPrimitives.isPrimitive(fcn.symbol)) {
                                val code = scalaPrimitives.getPrimitive(fcn.symbol, target_tree.tpe)
                                node.typ = "FcnCall" + (if (scalaPrimitives.isArrayOp(code)) {
                                    arrayExpr(code)
                                } else (args match {
                                    case Nil => unaryExpr(code)
                                    case right :: Nil => binaryExpr(code)
                                    case _ => assertFalse("bad argument list for arithmetic op",
                                        target_tree, code : java.lang.Integer, args)
                                }))
                            } else if (fcn.symbol.isStaticMember) {
                                node.typ = "StaticFcnCall"
                            } else if (fcn.symbol.isClassConstructor) {
                                node.typ = "ClassConstructorCall"
                            } else {
                                node.typ = "FcnCall"
                            }
                        case TypeApply(fcn, args) =>
                            node.typ = "FcnCallTypeApply"
                        case other =>
                            not_implemented("fcn call " + other, "type", fcn.getClass)
                            not_implemented("fcn call type " + other)
                    }
                }

            // already grabbed the symbol; what else matters?
            case LabelDef(name, params, rhs) => ()

            case Try(block, catches, finalizer) =>
                subtree("Try", block)
                subchain("Catch", catches)
                subtree("Finally", finalizer)



            // === expressions ===

            case ArrayValue(elemtpt, elems) =>
                subchain("Value", elems)

            case Ident(name) => ()
//                 symlink("Var", tree.symbol)

            case Select(qualifier, name) =>
                if (tree.symbol.isModule) {
                    // FIXME -- not quite...
                    node.typ = "QualifiedClass"
//                     symlink("Class", tree.symbol)
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

            case Throw(expr) =>
                subtree("Expr", expr)

            case Literal(Constant(value)) =>
                node.attrs.append( ("value", if (value == null) "null" else value.toString) )
                value match {
                    case v : Boolean => node.typ = "BooleanConstant"
                    case v : Char => node.typ = "CharConstant"
                    case v : Float => node.typ = "FloatConstant"
                    case v : Short => node.typ = "ShortConstant"
                    case v : Int => node.typ = "IntConstant"
                    case v : Long => node.typ = "LongConstant"
                    case v : String => node.typ = "StringConstant"
                    case () => node.typ = "UnitConstant"
                    case null => node.typ = "NullTypeConstant"
                    case r : AnyRef =>
                        not_implemented("scala constant literal", r.getClass().getName(),  value.toString)
                    case _ => if (value == null) {
                        node.typ = "NullPointer"
                    } else {
                        not_implemented("scala constant literal", value.toString)
                    }
                }

            case EmptyTree => ()

            case Bind(name, body) =>
                subtree("Body", body)

            // === branching ===

            case If(cond, thenstmt, elsestmt) =>
                subtree("Cond", cond)
                subtree("Then", thenstmt)
                subtree("Else", elsestmt)

            case Match(selector, cases) =>
                subtree("Target", selector)
                subchain("Case", cases)

            case CaseDef(pat, guard, body) =>
                subtree("Pattern", pat)
                subtree("Guard", guard)
                subtree("Body", body)

            case Typed(expr, typ) =>
                subtree("Expression", expr)
                symlink("Type", typ.symbol)



            // === statements ===

            case Assign(lhs, rhs) =>
                subtree("lhs", lhs)
                subtree("rhs", rhs)

            case Block(stmt_arr, expr) =>
                subchain("BlockStmt", stmt_arr)
                subtree("BlockExpr", expr)

            case ValDef(mods, name, typ, rhs) =>
                subtree("VarDeclRhs", rhs)

            case Return(expr) =>
                subtree("Expr", expr)

            case _ =>
                not_implemented("ERROR -- Scala construct not supported:", tree.getClass)
        }
        node
    }
}
