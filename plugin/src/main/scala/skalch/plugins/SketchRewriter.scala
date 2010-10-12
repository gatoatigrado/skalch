package skalch.plugins

import java.lang.Integer
import java.io.{File, FileInputStream, FileOutputStream}

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import scala.tools.nsc
import nsc._
import nsc.plugins.{Plugin, PluginComponent}
import nsc.io.{AbstractFile, PlainFile}
import nsc.util.{Position, FakePos, OffsetPosition, RangePosition}
import nsc.symtab.Flags._

import ScalaDebugOut._
import net.sourceforge.gxl._

/**
 * Scala plugin for sketching support. Generates a GXL file, executes a
 * script, and stores the result of the script in the header of the GXL file.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
class ExternalGxlTransformer(val global: Global) extends Plugin {
    import global._

    val name = "sketchrewriter"
    val gxl_extension = ".ast.gxl"
    val description = "Dumps a Scala AST to GXL for further processing"
    val components = List(TypeAnnotationComponent, GxlGeneratorComponent)

    val script_override = System.getenv().get("GRGEN_TRANSLATE")
    val script = if (script_override != null && (new File(script_override)).exists())
        script_override else "src/main/grgen/transform_sketch.py"

    val annotations = new HashMap[String, List[AnnotationInfo]]()

    object TypeAnnotationComponent extends PluginComponent {
        val global : ExternalGxlTransformer.this.global.type =
            ExternalGxlTransformer.this.global
        var symbolNameCtr = 0
        val runsAfter = List("typer")
        override val runsBefore = List("superaccessors")
        val phaseName = "grab_static_type_annotations"
        def newPhase(prev: Phase) = new GrabAnnotations(prev)

        class GrabAnnotations(prev: Phase) extends StdPhase(prev) {
            def apply(comp_unit : CompilationUnit) = {
                comp_unit.body = (new CreateTmpvarsForTyped()).transform(comp_unit.body)
                (new AnnotTf()).transform(comp_unit.body)
            }

            /** formatting */
            var ident = 0
            var currMethod : Symbol = null

            class CreateTmpvarsForTyped() extends Transformer {
                def createBlockForTyped(expr:Tree, tpt:Tree) = {
                    val name = newTermName("__skalch_internal_typed_tmp" + symbolNameCtr)
                    symbolNameCtr += 1
                    val sym = expr.tpe.typeSymbol.newValue(NoPosition, name).setInfo(tpt.tpe)
                    sym.owner = currMethod

                    // println("symbol annotations: " + sym.tpe.annotations)
                    // println("type annotations: " + tpt.tpe.annotations)
                    // println("normalized annotations: " + sym.tpe.normalize.annotations)

                    // core AST nodes
                    val newtyped = Typed(expr, tpt)
                    val vd = ValDef(Modifiers(LOCAL), name, tpt, newtyped)
                    val ident = Ident(name)
                    val replacement_block = Block(List(vd), ident)

                    // set symbols
                    vd.symbol = sym
                    ident.symbol = sym

                    // set types
                    vd.tpe = NoType
                    ident.tpe = tpt.tpe
                    newtyped.tpe = tpt.tpe
                    replacement_block.tpe = tpt.tpe

                    replacement_block
                }

                override def transform(tree : Tree) = tree match {
                    case x : DefDef => currMethod = x.symbol; super.transform(x)
                    case Typed(expr, tpt) =>
                        createBlockForTyped(super.transform(expr), super.transform(tpt))
                    case _ => super.transform(tree)
                }
            }

            class AnnotTf() extends Transformer {
                override def transform(tree : Tree) = {
                    val sym = tree.symbol

                    // def print_indented(text:String) = { for (v <- 0 to ident) print(" "); println(text) }
                    // def print_subident(text:String) = print_indented("- " + text)

                    if ((sym != null) && (sym != NoSymbol)) {
                    //--------------------------------------------------
                    // println("annottf visiting " + sym.name.toString())
                    //-------------------------------------------------- 
                        (sym.annotations ::: sym.tpe.normalize.annotations) match {
                            case Nil => ()
                            case lst => annotations.put(sym.fullName, lst)
                        }
                    }

                    ident += 4
                    val rv = super.transform(tree)
                    ident -= 4
                    rv
                }
            }
        }
    }

    /**
     * Generate a GXL representation of the sketch.
     */
    object GxlGeneratorComponent extends PluginComponent {
        val global : ExternalGxlTransformer.this.global.type =
            ExternalGxlTransformer.this.global
        val runsAfter = List("jvm")
//         override val runsBefore = List("jvm")
        val phaseName = "sketch_static_ast_gen"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        object gxl_node_map extends {
            val _glbl : ExternalGxlTransformer.this.global.type =
                ExternalGxlTransformer.this.global
        } with ScalaGxlNodeMap

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def run {
                scalaPrimitives.init
                super.run
                gxl_node_map.printNewNodes()
            }

            def apply(comp_unit : CompilationUnit) {
                val gxlout = new File(comp_unit.source.file.path + gxl_extension)
                val ast_gen = new GxlAstGenerator(comp_unit, gxlout)
                gxl_node_map.sourceFile = comp_unit.source.file.path
                ast_gen.transform(comp_unit.body)
//                 (new CheckVisited(gxl_node_map.visited)).transform(comp_unit.body)
            }

            class GxlAstGenerator(comp_unit : CompilationUnit, gxlout : File)
                extends Transformer
            {
                gxl_node_map.visited = new HashSet[Tree]()
                var root : gxl_node_map.GrNode = null

                override def transform(tree : Tree) : Tree = {
                    import GxlViews.{xmlwrapper, arrwrapper, nodewrapper}

                    val rcurl = getClass().getResource("/skalch/plugins/type_graph.gxl")
                    assert(rcurl != null, "resource type_graph.gxl not in jar package!")
                    val gxldoc = new GXLDocument(rcurl)
                    val gxlroot = gxldoc.getDocumentElement()
                    val graphs : Array[GXLGraph] = (for (i <- 0 until gxlroot.getGraphCount)
                        yield gxlroot.getGraphAt(i)).toArray
                    val typ_graph = graphs.get_by_attr("id" -> "SCE_ScalaAstModel")
                    val def_graph = graphs.get_by_attr("id" -> "DefaultGraph").asInstanceOf[GXLGraph]

                    // gxl type graph edges are really nodes
                    // kinda ugly but not hard to fix
                    val node_types = typ_graph.getnodes()
                    val real_nodes = node_types.filtertype(
                        "http://www.gupro.de/GXL/gxl-1.0.gxl#NodeClass")
                    val edges = node_types.filtertype(
                        "http://www.gupro.de/GXL/gxl-1.0.gxl#EdgeClass")

                    gxl_node_map.node_ids = real_nodes map (_.getAttribute("id"))
                    gxl_node_map.edge_ids = edges map (_.getAttribute("id"))

                    // FIXME -- move up when GXL library bugs are fixed
                    gxl_node_map.annotations.clear()
                    gxl_node_map.set_annotation_info(annotations)

                    root = gxl_node_map.getGxlAST(tree)

                    (new gxl_node_map.GxlOutput(def_graph)).outputGraph(root)
                    println("writing " + gxlout)
                    gxldoc.write(gxlout)

                    // FIXME -- remove this when bugs in the GXL library are fixed
                    gxl_node_map.sym_to_gr_map.clear()

                    tree
                }
            }

            /**
             * Make sure we're handling all of the nodes in the Scala AST.
             */
            class CheckVisited(transformed : HashSet[Tree]) extends Transformer {
                var parentCls : String = null
                override def transform(tree : Tree) : Tree = tree match {
                    case New(tpt) => tree
                    case TypeTree() => tree
                    case _ =>
                        if (!transformed.contains(tree)) {
                            println("WARNING - didn't transform node    " + tree.getClass() +
                                "; parent class " + parentCls)
                            println(tree.toString + "\n")
                        }
                        val prev_parent : String = parentCls
                        parentCls = tree.getClass().toString
                        val rv = super.transform(tree)
                        parentCls = prev_parent
                        rv
                }
            }
        }
    }
}
