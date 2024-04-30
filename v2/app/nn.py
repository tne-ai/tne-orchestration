"""
nn stands for "not None". The purpose of this module is to reduce
the amount of not None assertion boilerplate forced by type checking,
making initial development more rapid, while at the same time flagging
code that needs evaluation for possibly more robust error handling.

Using a type checker with multi-level Optional references like so:
    class Foo:
        text: Optional[str] = None
    class Bar:
        foo: Optional[Foo] = None

Where we have a bar typed as optional:
    bar: Optional[Bar]

Forces boilerplate patterns like this:
    assert bar
    assert bar.foo
    assert bar.foo.text
    text1: str = bar.foo.text

IF you have very short field names, it shortens to:
    assert bar and bar.foo and bar.foo.text
    text2: str = bar.foo.text

That breaks down with longer names:
    assert something_bar and something_bar.something_foo and something_bar.something_foo.something_text
    text2: str = something_bar.something_foo.something_text

But the below nn function allows this single-line pattern:
    text3: str = nn(nn(nn(something_bar).something_foo).something_text)

This is an ugly hack for rapid dev while maintaining some typing.

REMINDER: The use of nn also flags code which might need to be made
more robust with proper checking for None and error handling
rather than just asserting.
"""

from typing import Optional, TypeVar

_T = TypeVar("_T")


def nn(v: Optional[_T]) -> _T:
    "Assert given Optional value is not None and return non-Optional value."
    assert v is not None
    return v
