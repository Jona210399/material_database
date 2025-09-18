import re
import textwrap
import warnings
from collections import deque
from typing import Any

from typing_extensions import Self


class CifBlock:
    """
    Object for storing CIF data. All data is stored in a single dictionary.
    Data inside loops are stored in lists in the data dictionary, and
    information on which keys are grouped together are stored in the loops
    attribute.
    """

    max_len = 70  # not quite 80 so we can deal with semicolons and things

    def __init__(
        self,
        data: dict,
        loops: list[list[str]],
        header: str,
    ) -> None:
        """
        Args:
            data: dict of data to go into the CIF. Values should be convertible to string,
                or lists of these if the key is in a loop
            loops: list of lists of keys, grouped by which loop they should appear in
            header: name of the block (appears after the data_ on the first line).
        """
        self.loops = loops
        self.data = data
        # AJ (@computron) says: CIF Block names can't be more than 75 characters or you get an Exception
        self.header = header[:74]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.loops == other.loops
            and self.data == other.data
            and self.header == other.header
        )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def __str__(self) -> str:
        """Get the CIF string for the data block."""
        out = [f"data_{self.header}"]
        keys = list(self.data)
        written = []
        for key in keys:
            if key in written:
                continue
            for loop in self.loops:
                # search for a corresponding loop
                if key in loop:
                    out.append(self._loop_to_str(loop))
                    written.extend(loop)
                    break
            if key not in written:
                # key didn't belong to a loop
                val = self._format_field(self.data[key])
                if len(key) + len(val) + 3 < self.max_len:
                    out.append(f"{key}   {val}")
                else:
                    out.extend([key, val])
        return "\n".join(out)

    def _loop_to_str(self, loop: list[str]) -> str:
        """Convert a _loop block to string."""
        out = "loop_"
        for line in loop:
            out += "\n " + line

        for fields in zip(*(self.data[k] for k in loop), strict=True):
            line = "\n"
            for val in map(self._format_field, fields):
                if val[0] == ";":
                    out += f"{line}\n{val}"
                    line = "\n"
                elif len(line) + len(val) + 2 < self.max_len:
                    line += f"  {val}"
                else:
                    out += line
                    line = "\n  " + val
            out += line
        return out

    def _format_field(self, val: str) -> str:
        """Format field."""
        val = str(val).strip()

        if not val:
            return '""'

        # Wrap line if max length exceeded
        if len(val) > self.max_len:
            return f";\n{textwrap.fill(val, self.max_len)}\n;"

        # Add quotes if necessary
        if (
            (" " in val or val[0] == "_")
            and (val[0] != "'" or val[-1] != "'")
            and (val[0] != '"' or val[-1] != '"')
        ):
            quote = '"' if "'" in val else "'"
            val = quote + val + quote
        return val

    @classmethod
    def _process_string(cls, string: str) -> deque:
        """Process string to remove comments, empty lines and non-ASCII.
        Then break it into a stream of tokens.
        """
        # Remove comments
        string = re.sub(r"(\s|^)#.*$", "", string, flags=re.MULTILINE)

        # Remove empty lines
        string = re.sub(r"^\s*\n", "", string, flags=re.MULTILINE)

        # Remove non-ASCII
        string = string.encode("ascii", "ignore").decode("ascii")

        # Since line breaks in .cif files are mostly meaningless,
        # break up into a stream of tokens to parse, rejoin multiline
        # strings (between semicolons)
        deq: deque = deque()
        multiline: bool = False
        lines: list[str] = []

        # This regex splits on spaces, except when in quotes. Starting quotes must not be
        # preceded by non-whitespace (these get eaten by the first expression). Ending
        # quotes must not be followed by non-whitespace.
        pattern = re.compile(r"""([^'"\s][\S]*)|'(.*?)'(?!\S)|"(.*?)"(?!\S)""")

        for line in string.splitlines():
            if multiline:
                if line.startswith(";"):
                    multiline = False
                    deq.append(("", "", "", " ".join(lines)))
                    lines = []
                    line = line[1:].strip()
                else:
                    lines.append(line)
                    continue

            if line.startswith(";"):
                multiline = True
                lines.append(line[1:].strip())
            else:
                for string in pattern.findall(line):
                    # Location of the data in string depends on whether it was quoted in the input
                    deq.append(tuple(string))
        return deq

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Read CifBlock from string.

        Args:
            string: String representation.

        Returns:
            CifBlock
        """
        deq: deque = cls._process_string(string)
        header: str = deq.popleft()[0][5:]
        data: dict = {}
        loops: list[list[str]] = []

        while deq:
            _str = deq.popleft()
            # CIF keys aren't in quotes, so show up as _str[0]
            if _str[0] == "_eof":
                break

            if _str[0].startswith("_"):
                try:
                    data[_str[0]] = "".join(deq.popleft())
                except IndexError:
                    data[_str[0]] = ""

            elif _str[0].startswith("loop_"):
                columns: list[str] = []
                items: list[str] = []
                while deq:
                    _str = deq[0]
                    if _str[0].startswith("loop_") or not _str[0].startswith("_"):
                        break
                    columns.append("".join(deq.popleft()))
                    data[columns[-1]] = []

                while deq:
                    _str = deq[0]
                    if _str[0].startswith(("loop_", "_")):
                        break
                    items.append("".join(deq.popleft()))

                n = len(items) // len(columns)
                if len(items) % n != 0:
                    raise ValueError(f"{len(items)=} is not a multiple of {n=}")
                loops.append(columns)
                for k, v in zip(columns * n, items, strict=True):
                    data[k].append(v.strip())

            elif issue := "".join(_str).strip():
                warnings.warn(
                    f"Possible issue in CIF file at line: {issue}", stacklevel=2
                )

        return cls(data, loops, header)
