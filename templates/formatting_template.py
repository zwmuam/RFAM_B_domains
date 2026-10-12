# fmt: off
"""
===============================================================================
PEP 8 Visual Indent Style Template (No Post-Bracket Newline)
===============================================================================

1. Visual Alignment (Delimiter-Anchored): Continuation lines inside
   parentheses `()`, brackets `[]`, and braces `{}` begin on the exact same
   line as the opening delimiter. All subsequent lines align vertically with
   the character column directly following that opening delimiter.
2. No Post-Bracket Newline: Unlike Black/Ruff hanging-indent styles that place
   a newline immediately after `(`, `[`, or `{`, Visual Indent places the
   first item immediately after the opening delimiter.
3. Vertical Density: Preserves vertical screen real estate while maintaining
   strict horizontal alignment across parameter lists and collection entries.
4. Formatting Constraints: Requires disabling automated opinionated formatters
   (e.g., Black/Ruff) or enclosing blocks with `# fmt: off` / `# fmt: on`.
5. Max Line Length: Strictly respects the 99-character limit.
===============================================================================
"""

# built-ins
import functools
import os
from typing import Any, Dict, List, Optional, Set, Tuple

# standard libraries
import numpy as np

# internal repository/package imports
from prompt_toolkit import prompt


# class template
class ExampleClass:
    """
    ExampleClass description (concept, purpose and function of the class).
    """

    def __init__(self, example_param1: str,
                 example_param2: Optional[int] = None) -> None:
        """
        Constructor description (initialization logic).

        :param example_param1: Description of example_param1.
        :param example_param2: Description of example_param2.
        :return: None
        """
        self.example_param1 = example_param1
        self.example_param2 = example_param2

    @classmethod
    def decorated_method_example(cls, example_param1: int,
                                 example_param2: int) -> int:
        """
        Decorated (classmethod) method description.

        :param example_param1: Description of example_param1.
        :param example_param2: Description of example_param2.
        :return: Description of return value.
        """
        return example_param1 + example_param2

    def example_method(self, example_param1: str) -> bool:
        """
        Method description.

        :param example_param1: Description of example_param1.
        :return: Description of return value.
        """
        return bool(example_param1)


# standalone function template
def example_standalone_function(example_param1: str,
                                example_param2: int = 0) -> bool:
    """
    Standalone function description.

    :param example_param1: Description of example_param1.
    :param example_param2: Description of example_param2.
    :return: Description of return value.
    """
    return len(example_param1) > example_param2


# Visual Indent Function Call Example
# Specification: Arguments begin on opening line; subsequent lines align with `example_param1`.
example_variable = example_standalone_function(example_param1="example_string",
                                               example_param2=42)


# =============================================================================
# DATA STRUCTURE EXAMPLES (SET, TUPLE, LIST, DICT)
# =============================================================================

# --- SET EXAMPLES ---

# Style Spec: Inline Set (Small number of short elements on a single line)
inline_set: Set[str] = {"item1", "item2", "item3"}

# Style Spec: Multiline Visual Indent Set (One element per line)
multiline_set_standard: Set[str] = {"example_item_1",
                                    "example_item_2",
                                    "example_item_3"}

# Style Spec: Multiline Packed Set (Multiple short elements per line)
multiline_set_packed: Set[str] = {"elem1", "elem2", "elem3", "elem4",
                                  "elem5", "elem6", "elem7", "elem8"}

# Style Spec: Multiline Visual Indent Set with Complex Expressions
multiline_set_complex: Set[str] = {example_standalone_function("item1"),
                                   example_standalone_function("item2")}


# --- TUPLE EXAMPLES ---

# Style Spec: Inline Tuple (a few short elements in a single line)
inline_tuple: Tuple[int, int, int] = (1, 2, 3)

# Style Spec: Multiline Visual Indent Tuple (One element per line)
multiline_tuple_standard: Tuple[str, int, bool] = ("example_string",
                                                   100,
                                                   True)

# Style Spec: Multiline Packed Tuple (Multiple short elements per line)
multiline_tuple_packed: Tuple[int, ...] = (10, 20, 30, 40, 50, 60,
                                           70, 80, 90, 100)

# Style Spec: Multiline Visual Indent Tuple with Structured Pairs
multiline_tuple_complex: Tuple[Tuple[str, int], ...] = (("key1", 10),
                                                        ("key2", 20))


# --- LIST EXAMPLES ---

# Style Spec: Inline List (a few short elements in a single line)
inline_list: List[int] = [1, 2, 3, 4]

# Style Spec: Multiline Visual Indent List (One element per line)
multiline_list_standard: List[str] = ["example_element_1",
                                      "example_element_2",
                                      "example_element_3"]

# Style Spec: Multiline Packed List (Multiple short elements per line)
multiline_list_packed: List[int] = [10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25]

# Style Spec: Multiline Visual Indent List of Dictionaries
multiline_list_complex: List[Dict[str, Any]] = [{"key1": "val1"},
                                                {"key2": "val2"}]


# --- DICTIONARY EXAMPLES ---

# Style Spec: Inline Dictionary (a few short element pairs in a single line)
inline_dict: Dict[str, int] = {"key1": 1, "key2": 2}

# Style Spec: Multiline Visual Indent Dictionary (One key-value pair per line)
multiline_dict_standard: Dict[str, str] = {"key_one": "value_one",
                                           "key_two": "value_two"}

# Style Spec: Multiline Packed Dictionary (Multiple short key-value pairs per line)
multiline_dict_packed: Dict[str, int] = {"a": 1, "b": 2, "c": 3, "d": 4,
                                         "e": 5, "f": 6, "g": 7, "h": 8}

# Style Spec: Multiline Visual Indent Dictionary with Nested Structures
multiline_dict_complex: Dict[str, Any] = {"outer_key1": {"inner_key": "val"},
                                          "outer_key2": ["elem1",
                                                         "elem2"]}
# fmt: on
