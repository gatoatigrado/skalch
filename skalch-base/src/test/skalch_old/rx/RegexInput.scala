package test.skalch_old.rx

class RegexInput(val input : String, val result : Int) {
    def matched() : Boolean = (result != -1)
    override def toString() : String = "RegexInput[in = '" + input + "', matchlen = " + result
}
