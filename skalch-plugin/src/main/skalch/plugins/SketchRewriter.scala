package skalch.plugins

import java.lang.Integer
import java.io.{File, FileInputStream, FileOutputStream}

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import scala.tools.nsc
import nsc._
import nsc.plugins.{Plugin, PluginComponent}
import nsc.io.{AbstractFile, PlainFile}
import nsc.util.{FakePos, OffsetPosition, RangePosition}

import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/*
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.cli
*/

class SketchRewriter(val global: Global) extends Plugin {

    val name = "sketchrewriter"
    val fname_extension = ".hints.xml"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](ConstructRewriter,
        FileCopyComponent
        , SketchGeneratorComponent
        )
    var scalaFileMap = Map[Object, XmlDoc]()
    val fake_pos = FakePos("Inserted literal for call to sketch construct")

    case class ConstructFcn(val type_ : String, val uid : Int,
            val entire_pos : Object, val arg_pos : Object)
    {
        var parameter_type : String = "undefined"
    }
    case class XmlDoc(var cons_fcn_arr : List[ConstructFcn])

    object SketchXMLFormatter extends XmlFormatter {
        def formatObjectInner(x : Object) = x match {
            case XmlDoc(cons_fcn_arr) => ("document", List(),
                (for (construct <- cons_fcn_arr)
                    yield ("construct_" + construct.uid.toString, construct)).toList)
            case construct_fcn : ConstructFcn => (construct_fcn.type_,
                List(("uid", construct_fcn.uid.toString),
                    ("param_type", construct_fcn.parameter_type.toString)),
                List(("entire_pos", construct_fcn.entire_pos),
                    ("arg_pos", construct_fcn.arg_pos)))
            case rangepos : RangePosition => ("rangepos", List(),
                List(("start", rangepos.focusStart), ("end", rangepos.focusEnd)))
            case offsetpos : OffsetPosition => ("position", List(
                ("line", offsetpos.line.get.toString),
                ("column", offsetpos.column.get.toString)), List())
            case _ => ("unknown", List(("stringrep", x.toString)), List())
        }
    }

    /**
     * FIRST TASK - BEFORE THE TYPE SYSTEM.
     * Rewrite ??(arglist) to ??(uid, arglist)
     */
    private object ConstructRewriter extends SketchPluginComponent(global) {
        import global._
        val runsAfter = List[String]("parser");
        override val runsBefore = List[String]("namer")
        val phaseName = SketchRewriter.this.name
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def name = SketchRewriter.this.name
            var hintsSink: List[ConstructFcn] = null

            def apply(comp_unit: CompilationUnit) {
                hintsSink = List()
                comp_unit.body = CallTransformer.transform(comp_unit.body)
                if (!hintsSink.isEmpty) {
                    scalaFileMap += (comp_unit -> XmlDoc(hintsSink))
                }
                hintsSink = null
            }

            // Rewrite calls to ?? to include a call site specific uid
            object CallTransformer extends SketchTransformer {
                def zeroLenPosition(pos : Object) : RangePosition = pos match {
                    case rp : RangePosition => new RangePosition(
                        rp.source0, rp.end - 1, rp.end - 1, rp.end - 1)
                    case OffsetPosition(src, off) => {
                        println("note - range positions are not being used.")
                        new RangePosition( src, off - 1, off - 1, off - 1)
                    }
                    case _ => assert(false, "please enable range positions"); null
                }

                def transformSketchClass(clsdef : ClassDef) = null
                def transformSketchCall(tree : Apply, ct : CallType) = {
                    val uid = hintsSink.length
                    val uidLit = Literal(uid)
                    uidLit.setPos(fake_pos)
                    uidLit.setType(ConstantType(Constant(uid)))
                    val type_ = ct.cons_type match {
                        case ConstructType.Hole => "holeapply"
                        case ConstructType.Oracle => "oracleapply"
                    }
                    assert(hintsSink != null, "internal err - CallTransformer - hintsSink null");
                    // make a fake 0-length position
                    val arg_pos = (if (tree.args.length == 0) { zeroLenPosition(tree.pos) }
                        else { tree.args(0).pos })
                    hintsSink = hintsSink ::: List(new ConstructFcn(type_, uid, tree.pos, arg_pos))
                    treeCopy.Apply(tree, tree.fun, uidLit :: transformTrees(tree.args))
                }
            }
        }
    }

    /**
     * SECOND TASK - AFTER JVM.
     * Write out XML hints files -- info about constructs and source files.
     */
    private object FileCopyComponent extends SketchPluginComponent(global) {
        val runsAfter = List("jvm")
        val phaseName = "sketch_copy_src_desc"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            var processed = List[String]()
            var processed_no_holes = List[String]()

            def prefixLen(x0 : String, x1 : String) =
                (x0 zip x1).prefixLength((x : (Char, Char)) => x._1 == x._2)

            def stripPrefix(arr : List[String], other : List[String]) : List[String] = arr match {
                case head :: next :: tail =>
                    val longestPrefix = (for (v <- (arr ::: other)) yield prefixLen(v, head)).min
                    (for (v <- arr) yield v.substring(longestPrefix)).toList
                case _ => arr
            }

            def join_str(sep : String, arr : List[String]) = arr match {
                case head :: tail => (head /: tail)(_ + sep + _)
                case _ => ""
            }

            override def run() {
                currentRun.units foreach applyPhase
                val processed_noprefix = stripPrefix(processed, processed_no_holes)
                val processed_no_holes_noprefix = stripPrefix(processed_no_holes,
                    processed)
                println("sketchrewriter processed: " + join_str(", ", processed_noprefix))
                println("sketchrewriter processed, no holes: " + join_str(", ", processed_no_holes_noprefix))
            }

            def apply(comp_unit : CompilationUnit) {
                if (!scalaFileMap.keySet.contains(comp_unit)) {
                    processed_no_holes ::= comp_unit.source.file.path
                    return
                } else {
                    processed ::= comp_unit.source.file.path
                }

                val xmldoc = scalaFileMap(comp_unit)
                val out_dir = global.getFile(comp_unit.body.symbol, "")
                val out_dir_path = out_dir.getCanonicalPath.replaceAll("<empty>$", "")
                val copy_name = comp_unit.source.file.name + fname_extension
                val out_file = (new File(out_dir_path +
                    File.separator + copy_name)).getCanonicalPath

                var sketchClasses = ListBuffer[Symbol]()
                (new SketchDetector(sketchClasses, xmldoc)).transform(comp_unit.body)
                for (cls_sym <- sketchClasses) {
                    val cls_out_file = global.getFile(cls_sym, "") + ".info"
                    (new FileOutputStream(cls_out_file)).write("%s\n%s".format(
                        out_file, comp_unit.source.file.path).getBytes)
                }

                // NOTE - the tranformer now also adds info about which construct
                // was used.
                val xml : String = SketchXMLFormatter.formatXml(xmldoc)
                (new FileOutputStream(out_file)).write(xml.getBytes())
            }

            class SketchDetector(val sketchClasses : ListBuffer[Symbol],
                    val xmldoc : XmlDoc) extends SketchTransformer
            {
                def transformSketchClass(clsdef : ClassDef) = {
                    sketchClasses += clsdef.symbol
                    null
                }

                def transformSketchCall(tree : Apply, ct : CallType) = {
                    ct match {
                        case AssignedConstruct(cons_type, param_type) => {
                            val uid = tree.args(0) match {
                                case Literal(Constant(v : Int)) => v
                                case _ => -1
                            }
                            assert(param_type != null,
                                "please set annotations for call " + tree.toString())
                            xmldoc.cons_fcn_arr(uid).parameter_type = param_type
                        }
                        case _ => assert(false, "INTERNAL ERROR - NewConstruct after jvm")
                    }
                    null
                }
            }
        }
    }



    /**
     * THIRD TASK - ALSO AFTER JVM.
     * Generate the SKETCH AST and dump it via xstream.
     */
    private object SketchGeneratorComponent extends SketchPluginComponent(global) {
        val runsAfter = List("jvm")
        val phaseName = "sketch_static_ast_gen"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        /**
         * support early creation styles, where the sketch node is created
         * before its children are traversed. used for classes, which are
         * directly lowered into global functions.
         */
        class ContextInfo(val old : ContextInfo) {
            var curr_clazz : core.ScalaClass =
                if (old != null) old.curr_clazz else null
            var ident : String = if (old == null) "" else ("    " + old.ident)
        }

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            def apply(comp_unit : CompilationUnit) {
                val ast_gen = new SketchAstGenerator()
                ast_gen.transform(comp_unit.body)
                (new CheckVisited(ast_gen.visited)).transform(comp_unit.body)
            }

            class SketchAstGenerator() extends Transformer {
                val visited = new HashSet[Tree]()
                val symbol_type_map = new HashMap[String, nodes.Type]()
                var root : Object = null
                val name_string_factory = new SketchNames.NameStringFactory(false)

                import SketchNames.LogicalName
                import SketchNodes.{SketchNodeWrapper, SketchNodeList,
                    get_expr, get_stmt, get_param, get_expr_arr,
                    get_stmt_arr, get_param_arr, get_object_arr}

                /**
                 * The main recursive call to create SKETCH nodes.
                 */
                def getSketchAST(tree : Tree, info : ContextInfo) : SketchNodeWrapper = {
                    // TODO - fill in source values...
                    val ctx : nodes.FENode = null

                    if (visited != EmptyTree && visited.contains(tree)) {
                        DebugOut.print("warning: already visited tree", tree)
                    }
                    visited.add(tree)



                    // === accessor functions ===

                    def getname(elt : Object, sym : Symbol = tree.symbol) : String = {
                        name_string_factory.get_name(elt match {
                            case name : Name =>
                                def alternatives(idx : Int, verbosity : Int) = verbosity match {
                                    case 0 => name.toString
                                    case 1 => sym.owner.simpleName + "." + sym.simpleName
                                    case 2 => sym.fullNameString
                                    case 3 => name.toString + "_" + idx.toString
                                }
                                LogicalName(name.toString, sym.fullNameString,
                                    if (name.isTypeName) "type" else "term",
                                    _ + "_t", alternatives)
                            case t : Tree => subtree(t) match {
                                case subtree =>
                                    DebugOut.not_implemented("getname(tree)", tree, elt, subtree)
                                    null
                            }
                        })
                    }

                    def gettype_inner(tpe : Type) : nodes.Type = {
                        tpe.typeSymbol.fullNameString match {
                            case "scala.Int" => nodes.TypePrimitive.int32type
                            case "scala.Array" =>
                                tpe.typeArgs match {
                                    case Nil =>
                                        DebugOut.assertFalse("array with no type args")
                                        null
                                    case t :: Nil => new nodes.TypeArray(
                                        gettype_inner(t),
                                        new proxy.ScalaUnknownArrayLength(ctx))
                                    case lst =>
                                        DebugOut.assertFalse("array with many args " + lst)
                                        null
                                }
                            case "skalch.DynamicSketch$InputGenerator" =>
                                new skproxy.ScalaInputGenType()
                            case "skalch.DynamicSketch$HoleArray" =>
                                new skproxy.ScalaHoleArrayType()
                            case _ => DebugOut.not_implemented("gettype()",
                                    tpe, tpe.typeSymbol.fullNameString)
                                null
                        }
                    }
                    def gettype(tree : Tree) : nodes.Type = gettype_inner(tree.tpe)

                    def subtree(tree : Tree) = {
                        val rv = getSketchAST(tree, new ContextInfo(info))
                        DebugOut.print(info.ident + ".")
                        rv
                    }
                    def subarr(arr : List[Tree]) =
                        new SketchNodeList( (for (elt <- arr) yield subtree(elt)).toArray )



                    // === primary translation code ===
                    // most likely you want to fix a bug in the gettype functions 
                    val tree_str = tree.toString.replace("\n", " ")
                    DebugOut.print(info.ident +
                        "SKETCH AST translation for Scala tree", tree.getClass)
                    DebugOut.print(info.ident + tree_str.substring(0, Math.min(tree_str.length, 60)))
                    new SketchNodeWrapper(tree match {
                        // some code from GenICode.scala
                        // that file is a lot more complete though.
                        case Apply(fun, args) =>
                            new nodes.ExprFunCall(ctx, getname(fun), subarr(args))

                        case ArrayValue(elemtpt, elems) =>
                            val unused = subtree(elemtpt)
                            DebugOut.not_implemented("unused: elemtpt", unused)
                            new nodes.ExprArrayInit(ctx, subarr(elems))

                        case Assign(lhs, rhs) =>
                            new nodes.StmtAssign(ctx, subtree(lhs), subtree(rhs))

                        case Bind(name, body) =>
                            new stmts.ScalaBindStmt(
                                ctx, gettype(body), getname(name), subtree(body))

                        case Block(stmt_arr, expr) =>
                            new stmts.ScalaBlock(ctx, subarr(stmt_arr), subtree(expr))

                        case CaseDef(pat, guard, body) =>
                            new stmts.ScalaCaseStmt(
                                ctx, subtree(pat), subtree(guard), subtree(body))

                        case ClassDef(mods, name, tparams, impl) =>
                            val next_info = new ContextInfo(info)
                            next_info.curr_clazz = new core.ScalaClass(
                                ctx, getname(name), subarr(tparams))
                            symbol_type_map.put(tree.symbol.fullNameString, next_info.curr_clazz)
                            getSketchAST(impl, next_info)
                            next_info.curr_clazz

                        case DefDef(mods, name, tparams, vparamss, tpe, body) =>
                            // info.curr_clazz
                            val params = vparamss match {
                                case Nil => List[ValDef]()
                                case vparams :: Nil => vparams
                                case _ =>
                                    DebugOut.assertFalse("unknown defdef params", vparamss)
                                    null
                            }
                            // add the return node
                            val body_stmt = (subtree(body).node match {
                                case stmt : nodes.Statement => stmt
                                case expr : nodes.Expression => new nodes.StmtReturn(ctx, expr)
                            })
                            new core.ScalaClassFunction(ctx, nodes.Function.FUNC_PHASE,
                                getname(name), gettype(tpe), subarr(params), body_stmt)

                        case Ident(name) =>
                            new nodes.ExprVar(ctx, getname(name))

                        case If(cond, thenstmt, elsestmt) => new nodes.StmtIfThen(
                                ctx, subtree(cond), subtree(thenstmt), subtree(elsestmt))

                        case LabelDef(name, params, rhs) =>
                            new core.ScalaGotoLabel(
                                ctx, getname(name), subarr(params), subtree(rhs))

                        case Literal(value) =>
                            DebugOut.not_implemented("scala constant literal", value)
                            null
                            // new vars.ScalaConstantLiteral()

                        case Match(selector, cases) =>
                            new stmts.ScalaMatchStmt(
                                ctx, subtree(selector), subarr(cases))

                        case New(tpt : Tree) =>
                            new nodes.ExprNew(ctx, gettype(tpt))

                        case PackageDef(pid, stats) =>
                            println("NOTE - new package...")
                            DebugOut.print("stats", subarr(stats))
                            new proxy.ScalaPackageDef()

                        case Return(expr) =>
                            new nodes.StmtReturn(ctx, subtree(expr))

                        case Select(qualifier, name) =>
                            new nodes.ExprField(ctx, subtree(qualifier), getname(name))

                        case Super(qual, mix) =>
                            new vars.ScalaSuperRef(ctx, gettype(tree))

                        case Template(parents, self, body) =>
                            DebugOut.print("not visiting parents", parents)
                            for (sketch_node <- subarr(body).list) sketch_node match {
                                case f : core.ScalaClassFunction => ()
                                case _ =>
                                    DebugOut.not_implemented("element", "'" + sketch_node + "'",
                                        "in class body")
                                    ()
                            }
                            null

                        // qual may reference an outer class.
                        case This(qual) => new vars.ScalaThis(ctx,
                            symbol_type_map.get(tree.symbol.fullNameString).get)

                        case Throw(expr) =>
                            new stmts.ScalaThrow(ctx, subtree(expr))

                        case Try(block, catches, finalizer) =>
                            new stmts.ScalaTryCatchFinally(
                                ctx, subtree(block), subarr(catches), subtree(finalizer))

                        case TypeApply(fcn, args) =>
                            DebugOut.not_implemented("type apply", subtree(fcn), subarr(args))
                            new exprs.ScalaTypeApply(ctx, null, null)

                        case TypeTree() => gettype(tree)

                        case Typed(expr, typ) =>
                            new proxy.ScalaTypedExpression(
                                ctx, subtree(expr), gettype(typ))

                        case ValDef(mods, name, typ, rhs) => new nodes.StmtVarDecl(
                            ctx, gettype(typ), getname(name), subtree(rhs))

                        case EmptyTree =>
                            new proxy.ScalaEmptyExpression(ctx)

                        case _ =>
                            DebugOut.print("not matched " + tree)
                            null
                    })
                }

                override def transform(tree : Tree) : Tree = {
                    root = getSketchAST(tree, new ContextInfo(null))
                    tree
                }
            }

            /**
             * Make sure we're handling all of the nodes in the Scala AST.
             */
            class CheckVisited(transformed : HashSet[Tree]) extends Transformer {
                override def transform(tree : Tree) : Tree = {
                    if (!transformed.contains(tree)) {
                        //println("WARNING - didn't transform node\n    " + tree.toString())
                    }
                    super.transform(tree)
                }
            }
        }
    }
}
