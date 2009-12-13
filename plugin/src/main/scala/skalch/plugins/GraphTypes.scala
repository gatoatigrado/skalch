package skalch.plugins

import java.net.URI

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import scala.tools.nsc
import nsc._

import net.sourceforge.gxl._

/** create a level of abstraction above GXLNodes, which are only
 * output if they stay linked to a target. */
abstract class NodeFactory {
    val _glbl : Global
    import _glbl._
    import symtab.Flags

    val annotations = new HashMap[String, List[GrNode]]()

    def getGxlAST(tree : Tree) : GrNode

    /** print out all of the new nodes and edges created */
    val new_node_names = new HashSet[String]()
    var node_ids : Array[String] = null
    val new_edge_names = new HashSet[String]()
    var edge_ids : Array[String] = null

    def printNewNodes() {
        new_node_names foreach (x => println("node class %s;" format x))
        new_edge_names foreach (x => println("edge class %s;" format x))
        if (new_node_names.size + new_edge_names.size > 0) {
            assert(false, "insert new nodes into the graph");
        }
    }

    /** bookkeeping stuff */
    var sourceFile : String = null
    class SimplePosition(val line : Int, val col : Int)
    var id_ctr_ = 0
    def id_ctr() = {
        val rv = id_ctr_
        id_ctr_ += 1
        id_ctr_
    }

    class BoundNodeFcns(node : GrNode, clsname : String) {
        /** closure functions */
        def subtree(edge_typ : String, subtree : Tree) =
            GrEdge(node, edge_typ, getGxlAST(subtree))
        def subarr(edge_typ : String, arr : List[Tree]) =
            arr map (x => GrEdge(node, edge_typ, getGxlAST(x)))
        def subchain(edge_typ : String, arr : List[Tree]) = if (arr != Nil) {
            var nodes = arr map getGxlAST
            GrEdge(node, edge_typ + "Chain", nodes(0))
            GrEdge(node, edge_typ + "Last", nodes(nodes.length - 1))
            nodes.reduceLeft((x, y) => { GrEdge(x, edge_typ + "Next", y); y })
        } else GrEdge(node, edge_typ + "Chain", emptychainnode())
        def symlink(nme : String, sym : Symbol) = {
            if ((sym != null) && (sym != NoSymbol)) {
                GrEdge(node, nme + "Symbol", getsym(sym))
            } else (null)
        }
    }

    def get_annotation_node(info : AnnotationInfo) : GrNode = {
        var annot_node = new GrNode("Annotation", "annot_" + id_ctr())
        val node_fcns = new BoundNodeFcns(annot_node, "Annotation")
        import node_fcns._
        symlink("Annotation", info.atp.typeSymbol)
        subchain("AnnotationArgs", info.args)
        annot_node
    }

    def set_annotation_info(infos : HashMap[String, List[AnnotationInfo]]) {
        for ((k, arr) <- infos) {
            annotations.put(k, arr map (x => get_annotation_node(x)))
        }
    }

    /** symbol table map; shows usefulness of using graphs vs only trees
     * In lowering, will want to remove the symbol table and replace
     * it with more semantic edges, like "FunctionDefinition" or
     * "VariableDeclaration" or "EnclosingClassDefinition" */
    val sym_to_gr_map = new HashMap[Symbol, GrNode]()
    def getsym(sym : Symbol) : GrNode = sym_to_gr_map.get(sym) match {
        case None =>
            val node = new GrNode("Symbol", "symbol_" + sym.name + "_" + id_ctr())
            node.append_str_attr("symbolName", sym.name.toString().trim())
            node.append_str_attr("fullSymbolName", sym.fullNameString.trim())
            sym_to_gr_map.put(sym, node)

            val node_fcns = new BoundNodeFcns(node, "Symbol")
            import node_fcns._
            symlink("Type", sym.tpe.typeSymbol)

            def attr_edge(name : String) = GrEdge(node, name, node)
            if (sym != NoSymbol) {
                GrEdge(node, "SymbolOwner", getsym(sym.owner))

                // attribute edges
                if (sym.hasFlag(Flags.BRIDGE)) attr_edge("BridgeFcn")
                if (sym.isGetter) attr_edge("GetterFcn")
                if (sym.isSetter) attr_edge("SetterFcn")
                if (sym.isStaticMember) attr_edge("StaticMember")
                else if (sym.isMethod) attr_edge("ClsMethod")

                sym.tpe match {
                    case ClassInfoType(parents, decls, type_sym) =>
                        parents foreach ( (x : Type) => symlink("ParentType", x.typeSymbol) )

                    case TypeRef(pre, sym, args) =>
                        sym.tpe.parents.foreach( (x : Type) =>
                            symlink("ParentType", x.typeSymbol) )

                    case _ => ()
                }

                // static symbol annotations
                val symname = sym.fullNameString.replace("$", ".")
                annotations.get(symname) match {
                    case None => ()
                    case Some(lst) => lst foreach (x =>
                        GrEdge(node, "SymbolAnnotation", x))
                }
            }
            node
        case Some(node) => node
    }

    /** name is currently the unique name of the node; not to
     * be confused with e.g. the name of variables or classes. */
    class GrNode(var typ : String, var name : String) {
        type NNStr = String with NotNull
        type NNGxlValue = GXLValue with NotNull
        type NNTuple = Tuple2[NNStr, NNGxlValue] with NotNull

        /* lots of attribute-handling code */

        def get_nn_tuple(x : String, y : GXLValue) : NNTuple = {
            if ((x eq null) || (y eq null)) { assert(false,
                "get_nn_tuple: null inputs (%s, %s)" format (x, y)); }
            (x.asInstanceOf[NNStr] -> y.asInstanceOf[NNGxlValue]
                ).asInstanceOf[NNTuple]
        }

        def append_attr(x : String, y : GXLValue) { attrs.append(get_nn_tuple(x, y)) }

        def append_str_attr(x : String, y : String) {
            assert (! (y eq null), "y is null")
            append_attr(x, new GXLString(y))
        }

        def getattrlist() : ListBuffer[Tuple2[String, GXLValue]] = {
            val result = ListBuffer[Tuple2[String, GXLValue]]()
            attrs foreach ( (x : NNTuple) => result.append(
                x._1.asInstanceOf[String] -> x._2.asInstanceOf[GXLValue]) )
            return result
        }

        val attrs = new ListBuffer[NNTuple]()

        var use_default_type = true
        def set_type(typ : String, extend_ : String) {
            this.typ = typ
            this.use_default_type = false
            val typ_name = typ + (if (extend_ != null)
                " extends " + extend_ else " extends ScAstNode")
            if (!(node_ids contains typ)) {
                new_node_names.add(typ_name)
            }
        }
        /** register the type if nothing else was selected */
        def accept_type() { if (this.use_default_type) { set_type(typ, null) } }
        /** keep track of edges so only used nodes are output */
        val edges = new ListBuffer[GrEdge]()
        /** whether this node has been printed to the grshell script (or gxl) yet */
        var output : GXLNode = null

        def this(typ : String, name : String,
            start : SimplePosition, end : SimplePosition) =
        {
            this(typ, name)
            assert(sourceFile != null, "source file for node type %s null" format typ)
            attrs ++= ListBuffer(
                get_nn_tuple("sourceFile", new GXLString(sourceFile)),
                get_nn_tuple("startLine", new GXLInt(start.line)),
                get_nn_tuple("startCol", new GXLInt(start.col)),
                get_nn_tuple("endLine", new GXLInt(end.line)),
                get_nn_tuple("endCol", new GXLInt(end.col)) )
        }
    }

    def emptychainnode() = new GrNode("EmptyChain", "empty_chain_" + id_ctr())

    class GrEdge(val from : GrNode, val typ : String, val to : GrNode) {
        var output = false
    }

    /** probably override this later */
    def GrEdge(from : GrNode, typ : String, to : GrNode) = {
        val result = new GrEdge(from, typ, to)
        assert(edge_ids != null, "edge ids null")
        if (!(edge_ids contains typ)) {
            new_edge_names.add(typ)
            assert (!(typ contains "scala.tools.nsc.symtab"), "bad edge type")
        }
        from.edges.append(result)
        to.edges.append(result)
        result
    }

    class GxlOutput(var graph : GXLGraph) {
        val writtenNodes = new ListBuffer[GrNode]()
        val writtenEdges = new ListBuffer[GrEdge]()
        def outputGraph(node : GrNode) {
            outputGraphInner(node)
            for (n <- writtenNodes) n.output = null;
            for (e <- writtenEdges) e.output = false;
            writtenNodes.clear()
            writtenEdges.clear()
        }
        def outputGraphInner(node : GrNode) : GXLNode =
            if (node.output == null) {
                node.output = new GXLNode(node.name)
                node.output.setType(new URI("#" + node.typ))
                for ( (name, value) <- node.getattrlist() ) {
                    node.output.setAttr(name, value)
                }
                writtenNodes.append(node)
                graph.add(node.output)
                for (edge <- node.edges) {
                    if (!edge.output) {
                        edge.output = true
                        val from = outputGraphInner(edge.from)
                        val to = outputGraphInner(edge.to)
                        val gxledge = new GXLEdge(from, to)
                        gxledge.setType(new URI("#" + edge.typ))
                        writtenEdges.append(edge)
                        graph.add(gxledge)
                    }
                }
                node.output
            } else (node.output)
    }
}
