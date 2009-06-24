object PluginTest {
    def ??(y: Int) {
        Console.println("y: " + y)
        assert(false)
    }

    def ??(x: Int, y: Int) {
        Console.println("x: " + x)
        Console.println("y: " + y)
    }

    def main(args: Array[String]) {
        Console.println(??(2))
    }
}
