import pathlib

p = pathlib.Path("tests/test_inspect_tool.py")
t = p.read_text(encoding="utf-8")

# 把 ⚠ 换成 [WARN]
t = t.replace('assert tool.conclude([]).startswith("⚠")', 'assert tool.conclude([]).startswith("[WARN]")')

p.write_text(t, encoding="utf-8")
print("✅ Fixed test_inspect_tool.py")