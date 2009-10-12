package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.util.DebugOut
import sketch.compiler.ast.{base, core, scala => scast}

import scala.tools.nsc
import nsc._

import net.sourceforge.gxl._

/** map nodes */
abstract class NodeFactory() {
    val global : Global
    import global._

    println("TODO -- output symbol owner chain, for rewriting $this")

    /** rename some Scala AST nodes */
    val type_map = HashMap("Apply" -> "FcnCall")

    /** bookkeeping stuff */
    var sourceFile : String = null
    var start, end : Tuple2[Int, Int] = null
    var id_ctr_ = 0
    def id_ctr() = {
        val rv = id_ctr_
        id_ctr_ += 1
        id_ctr_
    }

    /** symbol table map; shows usefulness of using graphs vs only trees
     * In lowering, will want to remove the symbol table and replace
     * it with more semantic edges, like "FunctionDefinition" or
     * "VariableDeclaration" or "EnclosingClassDefinition" */
    val sym_to_gr_map = new HashMap[Symbol, GrNode]()
    def getsym(sym : Symbol) = sym_to_gr_map.get(sym) match {
        case None =>
            val node = new GrNode("Symbol", "symbol_" + sym.name + "_" + id_ctr())
            sym_to_gr_map.put(sym, node)
            node
        case Some(node) => node
    }

    /** name is currently the unique name of the node; not to
     * be confused with e.g. the name of variables or classes */
    class GrNode(var typ : String, var name : String) {
        /** keep track of edges so only used nodes are output */
        val edges = new ListBuffer[GrEdge]()
        /** whether this node has been printed to the grshell script (or gxl) yet */
        var output = false
        val attrs = new ListBuffer[Tuple2[String, String]]()
    }

    def GrNode(typ : String, name : String) = new GrNode(
        type_map.get(typ) match {
            case None => typ
            case Some(mapped) => mapped
        }, name)

    class GrEdge(val from : GrNode, val typ : String, val to : GrNode) {
        var output = false
    }

    /** probably override this later */
    def GrEdge(from : GrNode, typ : String, to : GrNode) = {
        val result = new GrEdge(from, typ, to)
        from.edges.append(result)
        to.edges.append(result)
        result
    }

    class GxlOutput(fname : java.io.File) {
        def outputGraph(node : GrNode) = {
            node.output = true
            for (edge <- node.edges) {
            }
        }
    }
}
