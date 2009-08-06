package skalch_old.simple;

import org.junit.Test;

import sketch.util.DebugOut;

public class SugaredJunitTest {
    @Test
    public void SugaredTest() throws Throwable {
        DebugOut.print("sugared test...");
        ScalaMainRunner.run("skalch_old.simple.SugaredTest", "--ui_no_gui");
    }
}
