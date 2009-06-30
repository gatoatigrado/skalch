package test

import ec.util.ThreadLocalMT
import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util._

/** N.B. compile me with optimise */
class RedBlackTreeSketch(val num_ops : Int,
    val num_tests : Int) extends DynamicSketch
{
    /** nodes only link to their children. the algorithm should
     * maintain a stack of parents when traversing the tree.
     */
    class TreeNode(var isBlack : Boolean, var value : Int) {
        var leftChild : TreeNode = null
        var rightChild : TreeNode = null
        /** assumes ident was already inserted. responsible for the newline */
        def formatTree(ident_ : String) : String = {
            val ident = ident_ + "    "
            val lcString = if (leftChild == null) "null\n" else leftChild.formatTree(ident)
            val rcString = if (rightChild == null) "null\n" else rightChild.formatTree(ident)
            toString() + "\n" + ident + "< " + lcString + ident + "> " + rcString
        }
        override def toString() = (if (isBlack) "B" else "R") + "(" + value + ")"

        // sketch-related
        def checkNumBlack() : Int = {
            val num_left = if (leftChild == null) 1 else leftChild.checkNumBlack()
            val num_right = if (rightChild == null) 1 else rightChild.checkNumBlack()
            synthAssertTerminal(num_left == num_right)
            num_left + num_right + (if (isBlack) 1 else 0)
        }
    }



    // array of nodes
    val all_nodes = (for (i <- 0 until num_ops) yield new TreeNode(false, 0)).toArray
    var num_active_nodes = 0
    var root : TreeNode = null



    // === non-sketched functions; mostly array maintenance ===
    def insertNode(value : Int) : TreeNode = {
        val result = all_nodes(num_active_nodes)
        if (root == null) {
            root = result
        }
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
    def printTree() {
        if (root == null) {
            skprint("null root")
        } else {
            skprint("=== tree ===\n" + root.formatTree(""))
        }
    }



    // === basic operations for the tree ===
    def recolorOp(node : TreeNode) = (node.isBlack = !!())
    def switchValueOp(n1 : TreeNode, n2 : TreeNode) {
        var tmp = n1.value
        n1.value = n2.value
        n2.value = tmp
    }
    // do a rotation
    def switchChildrenTuple(arr : TreeNode*) {
        val possibleChildren = new Array[TreeNode](3 * arr.length)
        for (i <- 0 until arr.length) {
            possibleChildren(3 * i) = arr(i).leftChild
            possibleChildren(3 * i + 1) = arr(i).rightChild
            possibleChildren(3 * i + 2) = arr(i)
        }
        for (i <- 0 until arr.length) {
            arr(i).leftChild = !!(possibleChildren)
            arr(i).rightChild = !!(possibleChildren)
        }
    }



    // === assertions ===
    /** first sketch: I want to specify the entire tree. per node is
     * possible but it would be maximally expressive if I could
     * place assertions on the tree only
     */
    def checkTree() {
        assert(root != null) // plain assert, this shouldn't happen at all
        root.checkNumBlack()
    }



    // === main functions ===
    def mainInsertRoutine(to_insert : TreeNode, parent : TreeNode,
        grandparent : TreeNode)
    {
        // go down the binary tree. using < vs. <= shouldn't matter (symmetry)
        if (parent.value > to_insert.value) {
            if (parent.rightChild == null) {
                parent.rightChild = to_insert
            } else {
                mainInsertRoutine(to_insert, parent.rightChild, parent)
            }
        } else if (parent.value <= to_insert.value) {
            // NOTE - I don't want to have to spell out this symmetry
            if (parent.leftChild == null) {
                parent.leftChild = to_insert
            } else {
                mainInsertRoutine(to_insert, parent.leftChild, parent)
            }
        }
    }
    /** one step of the reorganization procedure. returns true to recurse */
    def recursiveUpwardsStep(node : TreeNode, parent : TreeNode,
        grandparent : TreeNode) : Boolean =
    {
        !!()
    }


    def dysketch_main() = {
        num_active_nodes = 0
        root = null
        for (i <- 0 until num_ops) {
            val value = next_int_input()
            val next = insertNode(value)
            skprint("got value", value.toString)
            printTree()
        }
        !!()
        //true
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
