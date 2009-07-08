/** NOTE - CURRENTLY BROKEN
 * will work on it soon --ntung 2009-07-01
 */
package test

import scala.collection.mutable.{LinkedHashSet, ListBuffer}

import ec.util.ThreadLocalMT
import sketch.dyn.BackendOptions
import sketch.util._

/** N.B. compile me with optimise */
class RedBlackTreeSketch(val num_ops : Int,
    val num_tests : Int, val rand_input : Boolean)
    extends RBTreeSketchBase
{
    // array of nodes
    val all_nodes = (for (i <- 0 until num_ops) yield new RBTreeNode(false, 0)).toArray
    var num_active_nodes = 0
    var root : RBTreeNode = null



    // === non-sketched functions; mostly array maintenance ===
    def insertNode(value : Int) : RBTreeNode = {
        val result = all_nodes(num_active_nodes)
        num_active_nodes += 1
        // will get switched later.
        result.isBlack = true
        result.value = value
        // NOTE - almost forgot these. It would be nice if I didn't have to
        // worry about object allocation efficiency.
        result.leftChild = null
        result.rightChild = null
        result
    }



    // === basic operations for the tree ===
    def switchValueOp(n1 : RBTreeNode, n2 : RBTreeNode) {
        var tmp = n1.value
        n1.value = n2.value
        n2.value = tmp
    }
    // do a rotation. needed to move array "outside" for speed reasons
    val switchChildrenPossibilities = new LinkedHashSet[RBTreeNode]()
    val childrenPossibilitiesList = new ListBuffer[RBTreeNode]()
    def switchChildrenTuple(arr : RBTreeNode*) {
        switchChildrenPossibilities.clear()
        var num_children = 0
        for (node <- arr) {
            assert(node != null)
            switchChildrenPossibilities += node
            if (node.leftChild != null) {
                switchChildrenPossibilities += node.leftChild
                num_children += 1
            }
            if (node.rightChild != null) {
                switchChildrenPossibilities += node.rightChild
                num_children += 1
            }
        }
        childrenPossibilitiesList.clear()
        childrenPossibilitiesList.insertAll(0, switchChildrenPossibilities)
        var num_remaining = childrenPossibilitiesList.length
        var num_added = 0

        for (node <- childrenPossibilitiesList) {
            node.isBlack = !!()
        }

        def next_child() : RBTreeNode = {
            if (!!() && (num_remaining > 0)) {
                val node = childrenPossibilitiesList.remove(!!(num_remaining))
                num_remaining -= 1
                num_added += 1
                node
            } else {
                null
            }
        }

        for (node <- arr) {
            assert(node != null)
            node.leftChild = next_child()
            node.rightChild = next_child()
        }

        synthAssertTerminal(num_added == num_children)
    }



    // === assertions ===
    /** first sketch: I want to specify the entire tree. per node is
     * possible but it would be maximally expressive if I could
     * place assertions on the tree only
     */
    def checkTree() {
        assert(root != null) // plain assert, this shouldn't happen at all

        // asserts that don't rely upon the tree
        synthAssertTerminal(root.isBlack)
        root.checkIsTree()
        // skdprint("valid tree: " + root.formatTree(""))

        synthAssertTerminal(root.numNodes() == num_active_nodes)
        root.checkNumBlack()
        root.checkRedChildrenBlack()
    }



    // === main functions ===
    // ugh help me.... search not abstract enough
    /** returns the new child node (if the tree was rotated) */
    def recursiveInsertRoutine(to_insert__ : RBTreeNode, parent__ : RBTreeNode,
        grandparent : RBTreeNode) : RBTreeNode =
    {
        // always know to insert a black node as root
        var to_insert = to_insert__
        var parent = parent__
        if (root == null) {
            to_insert.isBlack = true
            return to_insert
        }
        var recolorInsert = true

        // go down the binary tree. using < vs. <= shouldn't matter (symmetry)
        if (parent.value < to_insert.value) {
            if (parent.rightChild == null) {
                parent.rightChild = to_insert
            } else {
                parent = recursiveInsertRoutine(to_insert, parent.rightChild, parent)
                to_insert = if (parent == null) { null } else { !!(parent.leftChild, parent.rightChild) }
                recolorInsert = false
            }
        } else {
            // NOTE - I don't want to have to spell out this symmetry
            if (parent.leftChild == null) {
                parent.leftChild = to_insert
            } else {
                parent = recursiveInsertRoutine(to_insert, parent.leftChild, parent)
                to_insert = if (parent == null) { null } else { !!(parent.leftChild, parent.rightChild) }
                recolorInsert = false
            }
        }

        if (parent == null) {
            // previous step signaled null
            null
        } else if (to_insert == null) {
            parent
        } else if (grandparent == null) {
            if (recolorInsert) {
                to_insert.isBlack = false
            }
            parent
        } else {
            recursiveUpwardsStep(to_insert, parent, grandparent)
            !!(grandparent, parent, to_insert)
        }
    }

    /** one step of the reorganization procedure. returns true to recurse */
    /*val switchChildrenArray = new Array[RBTreeNode](30)
    var switchChildrenArrayLength = 0
    def addNodes(node : RBTreeNode) {
    }*/
    def recursiveUpwardsStep(node : RBTreeNode, parent : RBTreeNode,
        grandparent : RBTreeNode)
    {
        // be stingy, use !! to start with only a few nodes

        /*node.isBlack = !!()
        parent.isBlack = !!()
        grandparent.isBlack = !!()*/
        switchChildrenTuple(node, parent, grandparent)
    }

    def mainInsertRoutine(to_insert : RBTreeNode, parent : RBTreeNode,
        grandparent : RBTreeNode)
    {
        val rv = recursiveInsertRoutine(to_insert, parent, grandparent)
        if (rv != null) {
            root = rv
            root.isBlack = true
        }
    }



    def dysketch_main() = {
        num_active_nodes = 0
        root = null
        for (i <- 0 until num_ops) {
            skdprint_loc("main loop - insert a new value")
            val value = next_int_input()
            skdprint("adding value " + value)
            mainInsertRoutine(insertNode(value), root, null)
            checkTree()
            skdprint(root.formatTree(""))
        }
        true
    }

    val test_generator = new TestGenerator {
        def set() {
            DebugOut.todo("more realistic tests")
            for (i <- 0 until num_ops) {
                // at least for now, synthesize insert before remove, hopefully fix a few
                // oracles so it can synthesize fast.
                val isInsert : Boolean = true
                    // RedBlackTreeTest.mt.get().nextBoolean()

                val value = if (rand_input) {
                    RedBlackTreeTest.mt.get().nextInt(100)
                } else {
                    i
                }
                put_default_input(value)
            }
        }
        def tests() { for (i <- 0 until num_tests) test_case() }
    }
}

object RedBlackTreeTest {
    val mt = new ThreadLocalMT()

    object TestOptions extends CliOptGroup {
        import java.lang.Integer
        add("--num_ops", 3 : Integer, "operations to process")
        add("--num_tests", 1 : Integer, "number of tests")
        add("--rand_input", "use random inputs (otherwise, debug inputs)")
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new RedBlackTreeSketch(
            opts.long_("num_ops").intValue,
            opts.long_("num_tests").intValue,
            opts.bool_("rand_input")))
    }
}
