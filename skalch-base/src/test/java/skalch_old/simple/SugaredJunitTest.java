package skalch_old.simple;

import static skalch_old.simple.ScalaMainRunner.ga_one_soln;
import static skalch_old.simple.ScalaMainRunner.run;

import org.junit.Test;

public class SugaredJunitTest {
    @Test
    public void SugaredTest() throws Throwable {
        run("skalch_old.simple.SugaredTest", ga_one_soln);
    }
}
