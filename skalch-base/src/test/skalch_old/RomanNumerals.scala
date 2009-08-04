package test.skalch_old

object RomanNumerals {

  abstract case class RomanNumeral() {
    def isI = false
    def isV = false
    def isX = false
    def isL = false
    def isC = false
    def isD = false
    def isM = false
    def value: Int
  }
  case class I() extends RomanNumeral {
    override def isI = true
    override def value: Int = 1
  }
  case class V() extends RomanNumeral {
    override def isV = true
    override def value: Int = 5
  }
  case class X() extends RomanNumeral {
    override def isX = true
    override def value: Int = 10
  }
  case class L() extends RomanNumeral {
    override def isL = true
    override def value: Int = 50
  }
  case class C() extends RomanNumeral {
    override def isC = true
    override def value: Int = 100
  }
  case class D() extends RomanNumeral {
    override def isD = true
    override def value: Int = 500
  }
  case class M() extends RomanNumeral {
    override def isM = true
    override def value: Int = 1000
  }

  def numberOfNumeral(numerals: List[RomanNumeral]): Int = {
    var num = 0
    var i = 0
    while (i < numerals.size) {
      val cur = numerals(i)
      val next = if (i + 1 < numerals.size) Some(numerals(i + 1)) else None
      val incr = cur match {
        case I() =>
          if (next.isDefined && next.get.isV) { i += 1; 4 } else if (next.isDefined && next.get.isX) { i += 1; 9 } else 1
        case V() => 5
        case X() =>
          if (next.isDefined && next.get.isL) { i += 1; 40 } else if (next.isDefined && next.get.isC) { i += 1; 90 } else 10
        case L() => 50
        case C() =>
          if (next.isDefined && next.get.isD) { i += 1; 400 } else if (next.isDefined && next.get.isM) { i += 1; 900 } else 100
        case D() => 500
        case M() => 1000
      }
      num += incr
      i += 1
    }
    num
  }

  def numeralOfString(s: String): List[RomanNumeral] = {
    import scala.collection.mutable.ListBuffer
    var buf = new ListBuffer[RomanNumeral]
    s foreach { c => {
      val cur = c match {
        case 'I' => I()
        case 'V' => V()
        case 'X' => X()
        case 'L' => L()
        case 'C' => C()
        case 'D' => D()
        case 'M' => M()
      }
      buf += cur
    }}
    buf.toList
  }

  /*def checkIsWellFormed(numeral: List[RomanNumeral]): Unit = {
    import java.lang.Math
    var i = 0
    var count = 0
    var prevprev: RomanNumeral = null
    var prev: RomanNumeral = null
    numeral foreach { cur => {
      if (i > 0) {
        assert(prev.value >= cur.value || (Math.log10(cur.value) - Math.log10(prev.value) <= 1 && (prevprev == null || prevprev.value < prev.value)))
        if (prev != cur)
          count = 0
      }
      prevprev = prev
      prev = cur
      count += 1
      assert(count < 4)
      i += 1
    }}
  }*/

}
