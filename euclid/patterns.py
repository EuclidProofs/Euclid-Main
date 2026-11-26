"""Bug-pattern scan for event-driven strategy code.

Reads the source for mistakes that are easy to make and hard to see in a
backtest. Heuristic: it can flag safe code and miss real bugs, so its findings
are hints with line numbers and never decide a verdict.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any

ORDER_METHODS = {"buy", "sell", "close", "order_target_size", "order_target_value",
                 "order_target_percent", "buy_bracket", "sell_bracket"}
PRICE_LINES = {"open", "high", "low", "close", "volume", "openinterest"}


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _places_order(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ORDER_METHODS and _is_self(n.func.value)
        for n in ast.walk(node)
    )


def _self_attributes_assigned(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets = [stmt.target]
    else:
        return []
    return [t.attr for t in targets if isinstance(t, ast.Attribute) and _is_self(t.value)]


def _blocks(node: ast.AST):
    """Yield every list of statements nested inside a node."""
    for field in ("body", "orelse", "finalbody"):
        block = getattr(node, field, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
            for stmt in block:
                yield from _blocks(stmt)
    for handler in getattr(node, "handlers", []):
        yield from _blocks(handler)


def _rooted_at_self_data(node: ast.AST) -> bool:
    """True for chains like self.data.close, self.data0.close, self.datas[0].close."""
    names = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        node = node.value
    return _is_self(node) and any(n.startswith("data") for n in names)


def scan(source: str, line_offset: int = 0) -> list[dict[str, Any]]:
    """Return hints: [{"line", "pattern", "code"}], sorted by line."""
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    hints: dict[tuple[int, str], dict[str, Any]] = {}

    def add(node: ast.AST, pattern: str) -> None:
        hints.setdefault((node.lineno, pattern), {
            "line": node.lineno + line_offset, "pattern": pattern,
            "code": lines[node.lineno - 1].strip()})

    for func in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for block in _blocks(func):
            order_placed = False
            for stmt in block:
                assigned = _self_attributes_assigned(stmt)
                if order_placed and assigned and not _places_order(stmt):
                    add(stmt, f"self.{assigned[0]} is updated right after an order is placed, "
                              "before knowing whether the order filled")
                if _places_order(stmt):
                    order_placed = True  # includes orders placed inside an if or loop

    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr in PRICE_LINES and _rooted_at_self_data(node.value)
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)
                and node.slice.value > 0):
            add(node, "a positive index reads a bar that has not happened yet")

    return sorted(hints.values(), key=lambda h: h["line"])
