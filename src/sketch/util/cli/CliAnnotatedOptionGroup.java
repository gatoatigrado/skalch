package sketch.util.cli;

import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.print;
import static sketch.util.DebugOut.print_exception;

import java.lang.reflect.Field;
import java.util.LinkedList;

public class CliAnnotatedOptionGroup extends CliOptionGroup {
    CliOptionResult lazy_results;
    LinkedList<Field> fields = new LinkedList<Field>();

    public CliAnnotatedOptionGroup(String prefix, String description) {
        super(prefix, description);
        for (Field field : this.getClass().getFields()) {
            CliParameter cli_annotation =
                    field.getAnnotation(CliParameter.class);
            if (cli_annotation != null) {
                try {
                    add("--" + field.getName(), field.get(this), cli_annotation
                            .help());
                    fields.add(field);
                } catch (Exception e) {
                    e.printStackTrace();
                    assertFalse("error accessing field", field);
                }
            }
        }
    }

    public void set_values() {
        // don't call this recursively
        if (!lazy_results.parser.set_on_parse.remove(this)) {
            assertFalse("call parse() on object", this);
        }
        print("annotated option group set_values()");
        try {
            for (Field field : fields) {
                if (!lazy_results.is_set(field.getName())) {
                    print("not set", field.getName());
                    continue;
                }
                if (field.getType() == Boolean.TYPE) {
                    print("set boolean", field.getName());
                    field.setBoolean(this, lazy_results.bool_(field.getName()));
                } else if (field.getType() == Integer.TYPE) {
                    print("set integer", field.getName());
                    field.setInt(this, (int) lazy_results
                            .long_(field.getName()));
                } else if (field.getType() == Long.TYPE) {
                    print("set long", field.getName());
                    field.setLong(this, lazy_results.long_(field.getName()));
                } else if (CliOptionType.class.isAssignableFrom(field
                        .getClass()))
                {
                    print("set other", field.getName());
                    field.set(this, lazy_results.other_type_(field.getName()));
                }
            }
        } catch (Exception e) {
            print_exception("set_values()", e);
        }
    }

    @Override
    public CliOptionResult parse(CliParser p) {
        lazy_results = super.parse(p);
        p.set_on_parse.add(this);
        return lazy_results;
    }
}
