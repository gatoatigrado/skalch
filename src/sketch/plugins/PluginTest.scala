object PluginTest {
    def ??(y: Int) = y * 100 + 100
    def ??(x: Int, y: Int) = x + y

    def main(args: Array[String]) {
        Console.println(??(1, 2))
        Console.println(??(2))
    }
}
