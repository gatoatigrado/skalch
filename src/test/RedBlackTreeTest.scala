/** NOTE - CURRENTLY BROKEN
 * will work on it soon --ntung 2009-07-01
 */
package test

import ec.util.ThreadLocalMT
import sketch.dyn.BackendOptions
import sketch.util._

/** N.B. compile me with optimise */
class RedBlackTreeSketch(val num_ops : Int,
    val num_tests : Int) extends RBTreeSketchBase
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
    def recolorOp(node : RBTreeNode) { if (node != null) { node.isBlack = !!() } }
    def switchValueOp(n1 : RBTreeNode, n2 : RBTreeNode) {
        var tmp = n1.value
        n1.value = n2.value
        n2.value = tmp
    }
    // do a rotation. needed to move array "outside" for speed reasons
    val possibleChildrenArr = new Array[RBTreeNode](30)
    def switchChildrenTuple(arr : Array[RBTreeNode], length : Int) {
        val num_possible_children = 3 * length
        for (i <- 0 until length) {
            if (arr(i) != null) {
                possibleChildrenArr(3 * i) = arr(i).leftChild
                possibleChildrenArr(3 * i + 1) = arr(i).rightChild
                possibleChildrenArr(3 * i + 2) = arr(i)
            } else {
                // need to clear nodes from last time
                possibleChildrenArr(3 * i) = null
                possibleChildrenArr(3 * i + 1) = null
                possibleChildrenArr(3 * i + 2) = null
            }
        }
        for (i <- 0 until length) {
            if (arr(i) != null) {
                arr(i).leftChild = possibleChildrenArr(!!(num_possible_children))
                arr(i).rightChild = possibleChildrenArr(!!(num_possible_children))
            }
        }
    }



    // === assertions ===
    /** first sketch: I want to specify the entire tree. per node is
     * possible but it would be maximally expressive if I could
     * place assertions on the tree only
     */
    def checkTree() {
        assert(root != null) // plain assert, this shouldn't happen at all
        if (!root.isBlack) {
            skdprint("root: " + root.toString)
        }

        // asserts that don't rely upon the tree
        synthAssertTerminal(root.isBlack)
        root.checkIsTree()
        skdprint("valid tree: " + root.formatTree(""))

        synthAssertTerminal(root.numNodes() == num_active_nodes)
        root.checkNumBlack()
        root.checkRedChildrenBlack()
    }



    // === main functions ===
    // ugh help me.... search not abstract enough
    /** returns the new child node (if the tree was rotated) */
    def mainInsertRoutine(to_insert__ : RBTreeNode, parent : RBTreeNode,
        grandparent : RBTreeNode) : RBTreeNode =
    {
        var to_insert = to_insert__
        if (root == null) {
            to_insert.isBlack = true
            root = to_insert
            return null
        }

        // go down the binary tree. using < vs. <= shouldn't matter (symmetry)
        if (parent.value < to_insert.value) {
            if (parent.rightChild == null) {
                parent.rightChild = to_insert
                skdprint("inserted to the right; tree before mutation: " + root.formatTree(""))
            } else {
                to_insert = mainInsertRoutine(to_insert, parent.rightChild, parent)
            }
        } else {
            // NOTE - I don't want to have to spell out this symmetry
            if (parent.leftChild == null) {
                parent.leftChild = to_insert
                skdprint("inserted to the left; tree before mutation: " + root.formatTree(""))
            } else {
                to_insert = mainInsertRoutine(to_insert, parent.leftChild, parent)
            }
        }
        if ( (to_insert == null) || !( !!() ) ) {
            null
        } else {
            recursiveUpwardsStep(to_insert, parent, grandparent)
            if (grandparent == null) {
                grandparent
            } else if (!!()) {
                grandparent.rightChild
            } else {
                grandparent.leftChild
            }
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

        val num_to_expand : Int = 10 - !!(4) + 1 // number of nodes to recolor
        skAddCost(num_to_expand)
        recolorOp(node)
        recolorOp(parent)
        recolorOp(grandparent)
        // switchChildrenTuple(SmallSubtree.arr, SmallSubtree.length)
    }



    def dysketch_main() = {
        num_active_nodes = 0
        root = null
        for (i <- 0 until num_ops) {
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

                // make the values human readable and easy to compare
                // val value : Int = RedBlackTreeTest.mt.get().nextInt(100)
                val value = i
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
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new RedBlackTreeSketch(
            opts.long_("num_ops").intValue,
            opts.long_("num_tests").intValue))
    }
}
